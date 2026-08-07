/**
 * action_channel.h — receives [thrustNorm, roll, pitch, yawRate] action
 * packets streamed up from the ground over the app-channel (see
 * crazyflie_cpp's Crazyflie::sendRaceAction, the transmitting half).
 *
 * Replaces the old obs_channel.c + onboard policyForward() pair: the
 * policy now runs on the ground station and only its OUTPUT crosses the
 * radio. See race_controller.c's docstring for why that is not a latency
 * regression (there was always exactly one radio hop in the loop, because
 * the observations themselves came from ground-side mocap).
 *
 * One action is one packet — 19 bytes, comfortably inside APPCHANNEL_MTU
 * (30) — so unlike a 3-chunk observation frame there is no reassembly and
 * no torn-frame failure mode. A dropped packet is simply a skipped update,
 * handled by race_controller.c's zero-order hold plus its staleness
 * fallback to PID.
 */
#ifndef ACTION_CHANNEL_H
#define ACTION_CHANNEL_H

#include <stdbool.h>
#include <stdint.h>

#define ACTION_CHANNEL_DIM 4

/**
 * Non-blocking: drains whatever app-channel packets are queued and writes
 * the NEWEST valid one into actionOut. Call once per control tick.
 *
 * Returns true exactly on a tick where at least one well-formed packet
 * arrived (actionOut is otherwise left untouched — the caller holds its
 * own last-known-good value). Values are NOT clipped here; the caller
 * clips to [-1, 1] before handing them to the mixer, matching training's
 * own env.step() clip.
 *
 * "Newest" is decided by wrapping-uint8 sequence comparison, not arrival
 * order, so a reordered pair inside a single drain can't apply the older
 * action. Duplicate/stale sequence numbers are discarded.
 */
bool actionChannelPoll(float actionOut[ACTION_CHANNEL_DIM]);

/**
 * Milliseconds between the last two accepted packets, measured from the
 * GROUND's own transmit timestamp rather than from arrival time — so this
 * reports the ground station's true send period, isolated from radio
 * jitter and from how often this gets polled.
 *
 * This is the measurement the 48 Hz question needs (see crazyflie_ros'
 * README: the stream rate is a placeholder, and /race_obs is currently
 * mocap-driven rather than rate-limited). Expect ~21 ms at 48 Hz. 0 until
 * two packets have been accepted.
 */
uint16_t actionChannelPeriodMs(void);

/** Reset sequence/period tracking — call from controllerOutOfTreeInit(). */
void actionChannelReset(void);

#endif // ACTION_CHANNEL_H
