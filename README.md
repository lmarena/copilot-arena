# Copilot Arena

Copilot Arena is an open source code AI coding assistant that provides paired autocomplete completions from different LLMs, which include state-of-the-art models like GPT-4o, Codestral, Llama-3.1 and more. Copilot Arena is **free** to use. Try it out to find out which models you like coding with!

Our goal is to evaluate which language models provide the best coding assistance. 

![Demo](assets/img/demo.gif)

## Installation Instructions

1. [Download the latest version here](assets/vsix/arena-0.2.1.vsix)
2. Open Visual Studio Code.
3. Go to the Extensions view (Ctrl+Shift+X).
4. Click on the "..." menu in the top-right corner of the Extensions view.
5. Select "Install from VSIX..." and choose the downloaded file.
6. Restart Visual Studio Code after installation.

If you've installed Copilot Arena before, please delete the ~/.code-arena folder to prevent any issues with new updates.

If installed successfully you will see Arena showing up on the bottom right corner of your window. 
When a completion is being generated, the check mark changes to a spinning circle.

## How do I use Copilot Arena?

### Understanding Your Completions

Copilot Arena adopts a slightly different user interface compared to a typical code completion.

1. Copilot Arena displays two completions, one on top of the other.
2. Copilot Arena repeats the same line prefix to keep the top and bottom outputs as similar as possible.

![Copilot Arena Example](assets/img/example.png)

**Note:** While we are still in the building process, you will also see two completion ids next to the ===== for debugging purposes. Please reference these ids when notifying us of any issues!

### Accepting Completions
Press ```Tab``` to accept the top completion and ```Shift-Tab``` to accept the bottom completion. You can also choose to accept neither completion and continue typing.

### Viewing your model preferences


![Preference Example](assets/img/model_pref_leaderboard.png)


When you accept a completion, you can find out which model generated it by checking the status bar. After twenty votes, you will unlock your personal leaderboard, where you can find out which models you prefer and their corresponding elo scores. Based on our experience, everyone will have slightly different model preferences. Stay tuned for a global leaderboard. 



## Privacy

Your privacy is important to us. The code in your current file is sent to our servers and sent to various API providers. By default, we do not collect your code for research purposes; we only log your vote for the leaderboard.
However, we log your code for the purposes of debugging. You can opt-out of this.

- To opt-in to data collection, please change `arena.codePrivacySettings` to `Research`.
- To disable logging entirely, please change `arena.codePrivacySettings` to `Private`. Opting-out means any bugs you encounter will be non-reproducable on our end.

You can find these settings by searching for "arena" in your vscode settings.

## Have a question or feedback?

Check out our [FAQs](FAQ.md) or submit an issue!