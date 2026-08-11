# app_race_policy

One firmware image, two ways to fly a `mjx-drone-trainer` checkpoint, both driven from the ground station — neither runs a policy onboard:

- **`action_type="attitude"` checkpoints**: sim-exact open-loop attitude mixing onboard, driven by policy actions streamed from the ground station. The Crazyflie converts a 4-float action straight to per-motor RPM, bypassing the stock attitude-rate PID entirely.
- **`action_type="mellinger"` checkpoints**: the vehicle's own stock Mellinger controller, driven by an ordinary attitude setpoint sent over the standard legacy RPYT commander — the same channel `crazyflie_ros` would use to fly this vehicle under plain PID. No action-channel involvement at all for this path.

Both are runtime-selectable (`ctrlRace.actChanEnable` / `ctrlRace.mellingerEnable`) on the SAME flashed binary — no reflash to switch which checkpoint type you're flying. See `src/race_controller.c`'s docstring for the full arbitration (`MODE_PID`/`MODE_STREAM`/`MODE_BENCH`/`MODE_MELLINGER`).

The policy itself used to run onboard; it no longer does, for either mode. There was always exactly one radio hop in the loop — observations come from ground-side mocap — so moving inference to the far side of that same hop costs no latency, and buys a simpler firmware, 3x less radio traffic, and free use of recurrent/asymmetric checkpoints (attitude-mixer mode specifically).

- **Policy:** runs on the **ground station**, not here. `crazyflie_ros` evaluates the [`mjx-drone-trainer`](https://github.com/Jirl-upenn/mjx-drone-trainer) checkpoint in JAX. For `action_type="attitude"`, it streams only the 4-float action output; for `action_type="mellinger"`, it sends a real attitude setpoint via the ordinary commander. Swapping checkpoints (even across action_types) is a ground-side file path — no export, no upload, no reflash.
- **Actions (attitude-mixer mode):** one 19-byte packet per control step via `Crazyflie::sendRaceAction`, reassembled by `src/action_channel.c`. A single packet is atomic, so there is no torn-frame failure mode.
- **Actuation (attitude-mixer mode)** — `src/mixer.c` emulates the policy output → per-motor RPMs from [`mjx-drone-trainer`](https://github.com/Jirl-upenn/mjx-drone-trainer).
- **Actuation (mellinger mode)** — `controllerMellingerFirmware()` (firmware's own stock controller, unmodified) on the standard setpoint.

## Expected Layout

Three sibling repos under one workspace directory. **Every path in this README is relative to this layout** — clone them side by side and the commands below work verbatim.

```
<workspace>/
├── crazyflie-firmware/                ← this repo
│   └── examples/app_race_policy/      ← you are here
├── crazyflie_ros/                     ← ROS 2 stack: mocap, radio driver, policy + action streaming
│   ├── bin/                           ← set_ctrl_race_params.py
│   └── tools/crazyflie_cpp/           ← Crazyflie::sendRaceAction
└── mjx-drone-trainer/                 ← training; checkpoints are read directly by crazyflie_ros
    └── runs/<task>/<run-name>/        ← checkpoints (params.pkl + config.json)
```

- Relative hops worth memorizing: from this directory, the workspace root is `../../..`, so `crazyflie_ros` is `../../../crazyflie_ros` and the trainer is `../../../mjx-drone-trainer`.
- The ground station loads `params.pkl` directly in JAX, so the flight laptop needs the trainer's runtime deps. Nothing is exported to C and nothing is uploaded to the vehicle.

## Control Modes

Since take-off, hover, and landing have a large sim-to-real gap, we maintain the original PID controller for these actions and only switch to NN command after stable hover is achieved. We also use the PID to prevent crashes due to stale / lost action packets.
| `mode` | Name | When | Drives motors via |
|---|---|---|---|
| 0 | PID | takeoff, hover, land, pre-race idle, stale/lost actions, `mellingerEnable=0` | `controllerPid()` |
| 1 | stream | action packets arriving *and* `actChanEnable=1` | `action_channel.c` → mixer |
| 2 | bench | `benchEnable=1` (overrides everything) | `ctrlRace.action0..3` → mixer |
| 3 | mellinger | `mellingerEnable=1` and no fresher action stream | `controllerMellingerFirmware()` on the standard setpoint |

Precedence (highest first): `benchEnable` → fresh action stream → `mellingerEnable` → PID. A fresh action stream always wins over `mellingerEnable` so a vehicle mid-race on the mixer path never silently drops to Mellinger if both happen to be set at once — in practice only one of `actChanEnable`/`mellingerEnable` should ever be set, matching whichever `action_type` the currently-loaded checkpoint declares.

## Build & Flash

Follow the [build docs](https://www.bitcraze.io/documentation/repository/crazyflie-firmware/master/building-and-flashing/build/) out of this directory rather than the project root:

```bash
export URI=radio://0/80/2M/E7E7E701B1     # your vehicle

cd <workspace>/crazyflie-firmware/examples/app_race_policy
make -j$(nproc)
make cload CLOAD_CMDS="-w $URI"
```
Notes:
- First build generates `build/.config` from `alldefconfig` merged with `app-config`; subsequent builds reuse it. `make clean` to reset.
- `make cload` warm-reboots the vehicle into its bootloader, flashes, and reboots. No physical button press needed.
- Config changes (`app-config`, Kconfig) need a `make clean` to take effect reliably.
- No weights are compiled in. A freshly flashed vehicle flies PID-only until a ground station starts streaming actions.

## Flying It

Flashing is where this repo's involvement ends. Everything after it — arming the stream or Mellinger mode, ROS bringup, and the takeoff/race/land sequence — belongs to the ground station:

**→ [`crazyflie_ros/README.md`](../../../crazyflie_ros/README.md) § Flight Sequence**

Come back here for the firmware-side reference below: what the `ctrlRace` params and logs mean, and what to check when a flight goes wrong.

## Params (`ctrlRace`)

Set with `set_ctrl_race_params.py --set NAME=VALUE`. All are runtime-writable — no reflash needed.

- **`actChanEnable`** (u8, default 0) — operator intent to hand over to the stream. Necessary, not sufficient. Attitude-mixer mode only; leave at 0 for a mellinger checkpoint.
- **`actStaleTicks`** (u16, default 50) — ms without an action packet before falling back to PID. Stabilizer ticks are 1 kHz, so this is milliseconds directly. Too tight → mode flaps mid-lap; too loose → a dropped link coasts further before recovering. Size it against the measured send rate (`ctrlRace.actPeriodMs`): at the intended 48 Hz a packet arrives every 20.8 ms, so 50 ms tolerates about two consecutive misses. That 48 Hz target is itself a placeholder — see `crazyflie_ros/README.md` § **Control-loop rate**. Attitude-mixer mode only.
- **`benchEnable`** (u8, default 0) — force bench mode: mixer driven by `action0..3` params, PID never runs. **Never set on a vehicle expected to fly under PID or Mellinger.**
- **`mellingerEnable`** (u8, default 0) — operator intent to route the standard setpoint through `controllerMellingerFirmware()` instead of `controllerPid()`. Necessary, not sufficient (a fresh action stream still wins — see Control Modes above). Mellinger mode only; leave at 0 for an attitude-mixer checkpoint. No separate staleness param: a lost/stale setpoint here is firmware's ordinary commander/supervisor watchdog's job, same as it would be under plain PID.
- **`hoverRpm` / `maxRpm` / `kf` / `differentialFrac`** — per-vehicle dynamics, attitude-mixer mode only. Defaults are `mjx-drone-trainer` sim nominals (mass 0.027 kg, `max_rpm` 21714, `kf` 3.16e-10, `differential_frac` 0.02), *not* calibrated values.
- **`action0..3`** — bench-test action stand-in, only read in `benchEnable=1`. Zeroed on init.

## Logs (`ctrlRace`)

- **`mode`** (u8) — 0=PID, 1=stream, 2=bench, 3=mellinger. **The ground truth for which controller produced the motor commands.** Pull this first when diagnosing a flight.
- **`actPackets`** (u32) — cumulative action packets accepted. Flat during a race means the uplink, not the policy, is the problem. Attitude-mixer mode only — always 0 under mellinger mode.
- **`actPeriodMs`** (u16) — ms between the last two packets, measured from the **ground's own** transmit timestamps, so it reports the sender's true period rather than radio jitter. This is the direct measurement of the stream rate that `crazyflie_ros`' README flags as a placeholder; expect ~21 ms at 48 Hz. Attitude-mixer mode only.
- **`rpm0..3`**, **`nf0..3`** — mixer output and the normalized per-motor forces handed to `powerDistributionForce()`. Attitude-mixer mode only — stale/frozen under mellinger mode, since that path never touches the mixer.

## Troubleshooting

- **Tips over on takeoff** — check `ctrlRace.mode` during takeoff. It must read 0. If it reads 1, action packets are arriving outside the racing state; if 2, `benchEnable` is set; if 3, `mellingerEnable` is set outside the racing state.
- **Attitude-mixer checkpoint never engages on `race`** — `actPackets` flat → uplink problem (radio, `crazyradio_driver`, namespace). `actPackets` climbing but `mode` still 0 → `actChanEnable=0`.
- **Mellinger checkpoint flies under PID instead** — check `mellingerEnable` is actually set to 1 (`crazyradio_driver.mellinger_enable` on the ground side); if it's already 1 and `mode` still reads 0, an action stream may also be arriving (a fresh stream always wins — see Control Modes) or a firmware issue is worth checking directly.
- **`mode` oscillates 0↔1 mid-lap** — send jitter exceeds `actStaleTicks`. Compare `actPeriodMs` against `actStaleTicks` and raise the latter; no reflash needed. Attitude-mixer mode only — mellinger mode has no equivalent stale-jitter mode flap, since it relies on firmware's own commander watchdog instead.
- **`actPeriodMs` far from 21 ms** — the ground station is not sending at the intended 48 Hz. Ground-side problem; see `crazyflie_ros`.

## Layout

- **`src/race_controller.c`** — `controllerOutOfTree()`, four-way mode arbitration (PID/stream/bench/mellinger), `ctrlRace` param/log groups.
- **`src/mixer.c`/`.h`** — action → per-motor RPM. Must stay a byte-exact port of the sim's mixer.
- **`src/action_channel.c`/`.h`** — single-packet action receive. Must agree byte-for-byte with `crazyflie_cpp`'s `sendRaceAction`.
- **`src/test_mixer_host.c`** — host-side numerical check of the mixer, built with plain gcc.
