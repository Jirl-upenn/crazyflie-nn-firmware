/**
 * race_controller.c — controllerOutOfTree() hook. Arbitrates between four
 * modes, per flight phase: the stock PID controller, the stock Mellinger
 * controller (driven by the SAME standard setpoint every other controller
 * receives — no separate channel), a ground-streamed open-loop mixer
 * action, and a bench-test stand-in for the mixer path.
 *
 * CONFIG_CONTROLLER_OOT=y makes this function the SOLE controller for the
 * entire flight (controller.c's autoselect chain picks ControllerTypeOot,
 * and controller() dispatches every tick through this one function
 * pointer) — PID/Lee/Mellinger are never reached via controller.c's own
 * dispatch on their own (their controllerFunctions[] entries still exist
 * and stabilizer.controller is technically still a writable param, but
 * writing it swaps the ACTIVE controller away from controllerOutOfTree()
 * entirely, discarding everything below — see MODE_MELLINGER instead,
 * which calls controllerMellingerFirmware() directly from inside this
 * function, keeping the arbitration/staleness/PID-fallback logic intact).
 * So anything this function does not handle itself, nothing else will:
 * takeoff, hover and land included. It therefore delegates to
 * controllerPid() for every phase except an actively-streaming race or
 * mellingerEnable being set.
 *
 * Getting this wrong is not subtle. Before this arbitration existed, this
 * function discarded `setpoint` outright and ran mixAttitudeRpm() on all
 * four phases: the workstation's CTBR takeoff setpoints were thrown away
 * on arrival and the vehicle instead flew open-loop constant hoverRpm with
 * no attitude feedback of any kind, which diverges immediately. That is a
 * tip-over on takeoff, every time.
 *
 * THE POLICY NO LONGER RUNS ONBOARD. It runs on the ground station, and
 * only its 4-float output crosses the radio (action_channel.c). This is
 * not a latency regression: there was always exactly ONE radio hop in the
 * loop, because the observations the policy consumes come from ground-side
 * mocap. Moving inference to the far side of that same hop trades an
 * ~1.75 ms software-fp16 forward pass on the STM32 for a sub-millisecond
 * one on a workstation. What it buys, concretely:
 *   - the action arrives pre-scaled in the mixer's own [-1, 1] units, so
 *     the uncalibrated action->deg/s constants the ground-side CTBR path
 *     needs (controller_policy_jax.py's max_*_rate_dps, documented there
 *     as placeholders) are not in the loop at all;
 *   - one packet per control step instead of a 3-chunk observation frame:
 *     ~3x less radio traffic and ~3x lower effective frame-loss rate, with
 *     torn frames structurally impossible;
 *   - recurrent/asymmetric/action-history checkpoints work unchanged,
 *     because the ground station runs the real training-repo network.
 * Onboard inference only wins on latency once observations are ALSO
 * computed onboard (see refs/howtomodel, which reads state->/sensors->
 * from the onboard EKF and never touches a ground station in its control
 * loop). That is not this architecture.
 *
 * During a race this path DOES bypass PID, going straight to per-motor RPM
 * via mixer.c and controlModeForce/normalizedForces (see
 * power_distribution_quadrotor.c's powerDistributionForce() — direct
 * per-motor thrust fraction, no mixing). That part is required, not
 * optional: every sim-trained checkpoint was trained against an OPEN LOOP
 * action->RPM mixer with no gyro feedback anywhere (see mixer.h's own
 * docstring) — routing a policy action through the real closed-loop PID
 * would substitute dynamics the policy never saw in training. The bypass
 * is correct; applying it to takeoff was the bug.
 *
 * Attitude (mixAttitudeRpm) is the SOLE control path here. mixRpmAction
 * itself stays in mixer.c/h — mixAttitudeRpm calls it internally for the
 * collective-thrust term, it's not rpm-mode-specific.
 *
 * `action[0..3]` is fed either by the ground stream (action_channel.c, in
 * MODE_STREAM) or, in MODE_BENCH, directly by the ctrlRace param group
 * itself — a bench-testing stand-in that lets the actuation path (mixer ->
 * kf -> normalizedForces -> real motors) be verified in isolation, with no
 * ground station in the loop at all. Both flow through the exact same
 * mixer/thrust code below; only how action0..3 get set differs.
 *
 * MODE_MELLINGER (ctrlRace.mellingerEnable) is for "mellinger"-action_type
 * checkpoints (crazyflow's own real physical [roll,pitch,yaw,thrust]
 * attitude convention, distinct from the open-loop mixer's
 * [thrustNorm,roll,pitch,yawRate]) — crazyflie_ros' JaxRacingPolicy sends
 * a real attitude setpoint via the ordinary legacy RPYT commander, exactly
 * like it would for stock Mellinger on a non-OOT build; this just calls
 * controllerMellingerFirmware() on that same setpoint instead of
 * controllerPid(), so ONE firmware image serves both action_types. No
 * separate staleness check is needed here (unlike MODE_STREAM's
 * actStaleTicks): a lost/stale setpoint is already firmware's generic
 * commander/supervisor watchdog's job, independent of which controller is
 * selected — see crtp_supervisor.c.
 */
#include "controller.h"
#include "controller_pid.h"
#include "controller_mellinger.h"
#include "power_distribution.h"
#include "mixer.h"
#include "action_channel.h"

#include "param.h"
#include "log.h"

// actChannelEnable is only the OPERATOR's intent to hand over to the
// ground-streamed policy. It is deliberately NOT sufficient on its own:
// the documented bringup sequence sets it while DISARMED, i.e. before
// takeoff, so it cannot distinguish "about to race" from "about to take
// off", and gating on it alone would put the stream in control of the
// takeoff. Action freshness is what actually separates the phases; see
// actStaleTicks.
static uint8_t actChannelEnable = 0;
static uint32_t actPacketsReceived = 0;
static uint16_t actPeriodMsLog = 0;

// THE phase gate, not merely a safety net. crazyflie_ros' controller_utils.py
// sends actions only while its FSM is in the 'racing' state — during
// takeoff/hover/land it sends ordinary CTBR setpoints and streams no
// actions at all. So "actions are arriving" is exactly equivalent to "the
// ground station believes we are racing", and it is the only signal
// available onboard that tracks the flight phase.
//
// This doubles as the link-loss failsafe: if the radio drops mid-lap the
// stream stops, this goes stale within actStaleTicks, and control reverts
// to PID rather than holding the last action forever (which would be a
// flyaway at whatever thrust was last commanded).
//
// stabilizerStep_t is a uint32_t tick count at 1000 Hz (stabilizer_types.h),
// so actStaleTicks is milliseconds. Size it against the real send rate,
// which ctrlRace.actPeriodMs now measures directly (expect ~21 ms at the
// intended 48 Hz). Too tight and the mode flaps mid-lap; too loose and a
// dropped link coasts further before recovering.
static uint16_t actStaleTicks = 50;
static stabilizerStep_t lastActTick = 0;
static bool actEverReceived = false;

// Bench-test escape hatch. Forces MODE_BENCH regardless of everything
// else: the mixer runs off the action0..3 params with no ground station
// in the loop, which is how the actuation path is verified on the bench.
// Never set this on a vehicle that is expected to fly under PID.
static uint8_t benchEnable = 0;

// Operator intent to route the standard setpoint (whatever the legacy
// RPYT commander last decoded — see this file's docstring) through
// controllerMellingerFirmware() instead of controllerPid(). Same
// "intent, not sufficient alone" framing as actChannelEnable, but there is
// no separate freshness check for it: an RPYT setpoint's own staleness is
// already firmware's generic commander/supervisor watchdog's job.
// Deliberately checked AFTER the stream check below, not before — a fresh
// action stream always wins, so a vehicle mid-race on the mixer path never
// silently drops to Mellinger if both happen to be set at once.
static uint8_t mellingerEnable = 0;

// Live mode, mirrored for ground-side logging: 0=PID, 1=stream, 2=bench,
// 3=mellinger. Log this alongside ctrlRace.actPackets when diagnosing a
// flight — it is the ground truth for which controller actually produced
// the motor commands on any given tick.
typedef enum {
  MODE_PID       = 0,
  MODE_STREAM    = 1,
  MODE_BENCH     = 2,
  MODE_MELLINGER = 3,
} RaceMode;
static uint8_t modeLog = MODE_PID;

// Per-vehicle dynamics constants — defaults are dmcdrones' MJXVectorAviary
// nominals (kf=3.16e-10, hover derived from mass=0.027kg at g=9.81,
// max_rpm=21714, differential_frac=0.02). Tunable via param so a real
// vehicle's calibrated values (from system ID, not simulation) can override
// these without a recompile/reflash — same mechanism ltfly's rlt.* params
// and this project's own mode-switching plan already rely on.
static float hoverRpm = 14475.81f;
static float maxRpm = 21714.0f;
static float kf = 3.16e-10f;
static float differentialFrac = 0.02f;

// Bench-testing action stand-in (see file docstring) — 0 everywhere is a
// safe default (thrustNorm=0 => hoverRpm on all motors, no differential —
// see mixAttitudeRpm). Never left at a non-hover value across a reset by
// controllerOutOfTreeInit() below, so a stale bench-test value from a
// previous flight can't silently carry into the next one.
static float action0 = 0.0f;
static float action1 = 0.0f;
static float action2 = 0.0f;
static float action3 = 0.0f;

// Exposed for ground-side bench-test logging/validation (see
// test_mixer_host.c for the offline numerical check this mirrors on-target).
static float rpmOut[4];
static float normalizedForcesOut[4];

void controllerOutOfTreeInit(void)
{
  action0 = 0.0f;
  action1 = 0.0f;
  action2 = 0.0f;
  action3 = 0.0f;
  lastActTick = 0;
  actEverReceived = false;
  modeLog = MODE_PID;
  actionChannelReset();
  // Required: nothing else initializes PID/Mellinger when
  // CONFIG_CONTROLLER_OOT=y, because controller.c only ever init()s the
  // one selected controller — which is this one. Safe to call here and
  // safe to call repeatedly in general; stabilizer.c does exactly that on
  // every runtime controller switch.
  controllerPidInit();
  controllerMellingerFirmwareInit();
}

bool controllerOutOfTreeTest(void)
{
  return true;
}

void controllerOutOfTree(control_t *control, const setpoint_t *setpoint,
                          const sensorData_t *sensors, const state_t *state,
                          const stabilizerStep_t stabilizerStep)
{
  // Every parameter is now genuinely used — setpoint/sensors/state all flow
  // through to controllerPid() below. The unused-parameter casts that used
  // to sit here were the visible symptom of the takeoff bug described in
  // this file's docstring; they are gone on purpose.

  // Drain the app-channel every tick regardless of mode. Doing this
  // outside the mode branch matters: it lets a packet arriving during PID
  // flight promote us into MODE_STREAM on this very tick, and it keeps the
  // queue from backing up with stale actions while we are not racing.
  if (actChannelEnable != 0) {
    float streamed[ACTION_CHANNEL_DIM];
    if (actionChannelPoll(streamed)) {
      // Clip to [-1, 1] — matches training-time env.step()'s own
      // unconditional clip (see mixer.h's docstring: both mixers assume a
      // pre-clipped input, they don't clip it themselves). This is the only
      // sanity check standing between an over-the-air packet and the
      // motors, so it is not optional.
      float a0 = streamed[0], a1 = streamed[1];
      float a2 = streamed[2], a3 = streamed[3];
      if (a0 < -1.0f) { a0 = -1.0f; } else if (a0 > 1.0f) { a0 = 1.0f; }
      if (a1 < -1.0f) { a1 = -1.0f; } else if (a1 > 1.0f) { a1 = 1.0f; }
      if (a2 < -1.0f) { a2 = -1.0f; } else if (a2 > 1.0f) { a2 = 1.0f; }
      if (a3 < -1.0f) { a3 = -1.0f; } else if (a3 > 1.0f) { a3 = 1.0f; }
      action0 = a0;
      action1 = a1;
      action2 = a2;
      action3 = a3;
      actPacketsReceived++;
      actPeriodMsLog = actionChannelPeriodMs();
      lastActTick = stabilizerStep;
      actEverReceived = true;
    }
    // No new packet this tick: the last action is held (zero-order hold)
    // for up to actStaleTicks, then we fall back to PID below.
  }

  // Unsigned subtraction on a monotonic uint32_t tick counter, so this is
  // wrap-safe. actEverReceived guards the boot case: without it, a
  // never-written lastActTick of 0 would read as perfectly fresh on tick 0
  // and hand a just-powered-on vehicle straight to the stream.
  bool actFresh = actEverReceived &&
                  ((stabilizerStep - lastActTick) <= (stabilizerStep_t)actStaleTicks);

  RaceMode mode = MODE_PID;
  if (benchEnable != 0) {
    mode = MODE_BENCH;
  } else if (actChannelEnable != 0 && actFresh) {
    mode = MODE_STREAM;
  } else if (mellingerEnable != 0) {
    mode = MODE_MELLINGER;
  }
  modeLog = (uint8_t)mode;

  if (mode == MODE_PID) {
    // Takeoff, hover, land, pre-race idle, and any loss of the action
    // stream all land here. controllerPid() writes control->controlMode
    // (controlModeLegacy) itself, so the mode alternation across a race
    // boundary is handled downstream by powerDistribution()'s own switch.
    //
    // Deliberately NOT re-initializing PID on the stream->PID edge: its
    // integrators sit frozen while the stream flies, so they resume at
    // their pre-race hover values. Zeroing them instead would drop the
    // z-integral that was compensating hover thrust bias, and the vehicle
    // would sag on handback before rebuilding it.
    controllerPid(control, setpoint, sensors, state, stabilizerStep);
    return;
  }

  if (mode == MODE_MELLINGER) {
    // Same setpoint every other mode/controller receives (built upstream
    // by the standard commander from crazyflie_ros' ordinary legacy RPYT
    // send — see this file's docstring) — no action-channel involvement
    // at all. controllerMellingerFirmware() writes control->controlMode
    // (controlModeLegacy) itself, same as controllerPid() above.
    controllerMellingerFirmware(control, setpoint, sensors, state, stabilizerStep);
    return;
  }

  float action[4] = {action0, action1, action2, action3};
  mixAttitudeRpm(action, hoverRpm, maxRpm, differentialFrac, rpmOut);

  // force_i = kf * rpm_i^2 (N) -> fraction of one motor's calibrated max
  // thrust. powerDistributionGetMaxThrust() is the TOTAL across all
  // motors (see its own docstring in power_distribution.h) — divide by
  // STABILIZER_NR_OF_MOTORS for the per-motor figure powerDistributionForce()
  // expects normalizedForces to be relative to. Read every call (not
  // cached) since it can change if idle-thrust/battery-compensation config
  // changes at runtime; the cost is negligible next to the physics step.
  float perMotorMaxThrust = powerDistributionGetMaxThrust() / STABILIZER_NR_OF_MOTORS;
  for (int i = 0; i < STABILIZER_NR_OF_MOTORS; i++) {
    float force = kf * rpmOut[i] * rpmOut[i];
    // Not clamped here: powerDistributionForce() (power_distribution_quadrotor.c)
    // already clips normalizedForces to [0, 1] itself before scaling to
    // UINT16_MAX — an out-of-range value here is handled downstream, not a
    // bug in this function specifically.
    normalizedForcesOut[i] = force / perMotorMaxThrust;
    control->normalizedForces[i] = normalizedForcesOut[i];
  }
  control->controlMode = controlModeForce;
}

PARAM_GROUP_START(ctrlRace)
PARAM_ADD(PARAM_UINT8, actChanEnable, &actChannelEnable)
PARAM_ADD(PARAM_UINT16, actStaleTicks, &actStaleTicks)
PARAM_ADD(PARAM_UINT8, benchEnable, &benchEnable)
PARAM_ADD(PARAM_UINT8, mellingerEnable, &mellingerEnable)
PARAM_ADD(PARAM_FLOAT, hoverRpm, &hoverRpm)
PARAM_ADD(PARAM_FLOAT, maxRpm, &maxRpm)
PARAM_ADD(PARAM_FLOAT, kf, &kf)
PARAM_ADD(PARAM_FLOAT, differentialFrac, &differentialFrac)
PARAM_ADD(PARAM_FLOAT, action0, &action0)
PARAM_ADD(PARAM_FLOAT, action1, &action1)
PARAM_ADD(PARAM_FLOAT, action2, &action2)
PARAM_ADD(PARAM_FLOAT, action3, &action3)
PARAM_GROUP_STOP(ctrlRace)

LOG_GROUP_START(ctrlRace)
LOG_ADD(LOG_FLOAT, rpm0, &rpmOut[0])
LOG_ADD(LOG_FLOAT, rpm1, &rpmOut[1])
LOG_ADD(LOG_FLOAT, rpm2, &rpmOut[2])
LOG_ADD(LOG_FLOAT, rpm3, &rpmOut[3])
LOG_ADD(LOG_FLOAT, nf0, &normalizedForcesOut[0])
LOG_ADD(LOG_FLOAT, nf1, &normalizedForcesOut[1])
LOG_ADD(LOG_FLOAT, nf2, &normalizedForcesOut[2])
LOG_ADD(LOG_FLOAT, nf3, &normalizedForcesOut[3])
LOG_ADD(LOG_UINT32, actPackets, &actPacketsReceived)
// 0=PID, 1=stream, 2=bench, 3=mellinger. Reports which controller actually
// drove the motors, not merely what the operator requested.
LOG_ADD(LOG_UINT8, mode, &modeLog)
// Ground-measured send period in ms (see action_channel.h) — this is the
// direct measurement of the stream rate that crazyflie_ros' README calls a
// placeholder. ~21 ms at the intended 48 Hz.
LOG_ADD(LOG_UINT16, actPeriodMs, &actPeriodMsLog)
LOG_GROUP_STOP(ctrlRace)
