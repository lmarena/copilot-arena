# FAQ

**Q: My tab isn't working / Copilot Arena doesn't work well with Copilot code completions!**

**A:** Yes, Copilot code completions don't work with Copilot Arena as (a) they both use the completions API and (b) they both require use of the tab key. You don't have to disable all of Copilot; instead, if you click on the copilot icon in the bottom right corner, you should be able to disable their completions feature only.

If you are using vim mode, try disabling vim mode (or only requesting completions in vim insert mode)

**Q: Copilot Arena is lagging my VSCode.**

**A:** Please restart VSCode; it should fix itself in less than a minute after you restart the application. This should be fixed in the latest version, so if you are still experiencing this issue

**Q: My completions are not being generated?**

**A:** Try a few times to ensure that it isn't due to cold start. Also, try restarting VSCode or triggering the suggestions manually.
Note that completions will not display if (a) you move away from the window or (b) the suggestions window (i.e. classic autocomplete) is showing.

You can also run the "Developer: Toggle Developer Tools" command to see debug logs. You should see logs like the following which indicate the completions are being generated.

Also, if you've installed Copilot Arena before and are encountering any issues, please delete the ~/.copilot-arena folder to prevent any issues with new updates.

**Q: What is the expected response time for a completion?**

**A:** Most completions should finish in around 1 second, although some may take up to 2 seconds. Some model APIs have high latencies and we are working with model providers to decrease latency.

**Q: How do I trigger the inline suggestion manually?**

**A:** There is a command called "Trigger Inline Suggestion" you can run. This is oftentimes much faster than waiting for an inline suggestion.

