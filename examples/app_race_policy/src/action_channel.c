#include <string.h>
#include <stdint.h>

#include "action_channel.h"
#include "app_channel.h"

// Must match crazyflie_cpp's Crazyflie::sendRaceAction byte for byte.
// 1 + 2 + 16 = 19 bytes, inside APPCHANNEL_MTU (30). Packed because the
// ground side writes a flat little-endian buffer with no padding; the
// members are read out via memcpy below rather than dereferenced in place,
// so the unaligned uint16/float members are safe on Cortex-M.
typedef struct {
  uint8_t  seq;        // wrapping counter, incremented once per ground control step
  uint16_t txTickMs;   // ground's own send timestamp, wrapping at 65536 ms
  float    action[ACTION_CHANNEL_DIM];
} __attribute__((packed)) ActionPacket;

static bool haveSeq = false;
static uint8_t lastSeq = 0;
static uint16_t lastTxTickMs = 0;
static uint16_t periodMs = 0;

void actionChannelReset(void)
{
  haveSeq = false;
  lastSeq = 0;
  lastTxTickMs = 0;
  periodMs = 0;
}

uint16_t actionChannelPeriodMs(void)
{
  return periodMs;
}

bool actionChannelPoll(float actionOut[ACTION_CHANNEL_DIM])
{
  ActionPacket pkt;
  bool accepted = false;
  uint8_t bestSeq = lastSeq;
  uint16_t bestTx = 0;
  float bestAction[ACTION_CHANNEL_DIM];

  // Drain everything queued rather than one packet per tick: the ground
  // sends faster than this is polled under some configurations, and a
  // backlog of stale actions must never be applied one-per-tick after the
  // fact.
  while (appchannelReceiveDataPacket(&pkt, sizeof(pkt), 0) == sizeof(pkt)) {
    if (haveSeq || accepted) {
      // Wrapping-safe "is newer": signed difference over uint8. Rejects
      // duplicates (diff == 0) and reordered older packets (diff < 0).
      int8_t diff = (int8_t)(pkt.seq - bestSeq);
      if (diff <= 0) {
        continue;
      }
    }
    bestSeq = pkt.seq;
    memcpy(&bestTx, &pkt.txTickMs, sizeof(bestTx));
    memcpy(bestAction, pkt.action, sizeof(bestAction));
    accepted = true;
  }

  if (!accepted) {
    return false;
  }

  // Ground-clock delta, so this reports the sender's period rather than
  // this tick's arrival jitter. Only meaningful once a previous packet has
  // been seen; wrapping subtraction is correct across the 65536 ms roll.
  if (haveSeq) {
    periodMs = (uint16_t)(bestTx - lastTxTickMs);
  }
  lastSeq = bestSeq;
  lastTxTickMs = bestTx;
  haveSeq = true;

  memcpy(actionOut, bestAction, sizeof(bestAction));
  return true;
}
