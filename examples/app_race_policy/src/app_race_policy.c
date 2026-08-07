/**
 * app_race_policy.c — app entry point.
 *
 * controllerOutOfTree() (race_controller.c) arbitrates per flight phase
 * between the stock PID controller and a ground-streamed policy action —
 * see that file's docstring. The app-channel receive (action_channel.c) is
 * driven from that controller; no network runs onboard any more. appMain()
 * itself has nothing to do — the FreeRTOS stabilizer task calls
 * controllerOutOfTree() on its own schedule, independent of this app task
 * — so it just idles.
 */
#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "RACEPOLICY"
#include "debug.h"

void appMain()
{
  DEBUG_PRINT("app_race_policy: mixer + controlModeForce wired, app-channel/policy not yet\n");
  while (1) {
    vTaskDelay(M2T(2000));
  }
}
