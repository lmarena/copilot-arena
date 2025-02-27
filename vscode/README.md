# Copilot Arena

[Check out our Github Repository!](https://github.com/lm-sys/copilot-arena)

Copilot Arena is an open source code AI coding assistant that provides paired autocomplete completions from different LLMs, which include state-of-the-art models like GPT-4o, Codestral, Llama-3.1 and more. Copilot Arena is **free** to use.

The goal of this project is to evaluate which language models provide the best coding assistance.

![Demo](https://raw.githubusercontent.com/lm-sys/copilot-arena/36377762f69231241dcff00817cb8801859612e7/assets/img/demo.gif)

### Maintainers
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/waynechi?style=flat-square&logo=x&label=Wayne%20Chi)](https://twitter.com/iamwaynechi)
[![GitHub](https://img.shields.io/badge/waynchi-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/waynchi)
[![Website](https://img.shields.io/badge/waynechi.com-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://www.waynechi.com/)

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/valeriechen_?style=flat-square&logo=x&label=Valerie%20Chen)](https://twitter.com/valeriechen_)
[![GitHub](https://img.shields.io/badge/valeriechen-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/valeriechen)
[![Website](https://img.shields.io/badge/valeriechen.github.io-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://valeriechen.github.io/)


## Read This To Get Started

Step 1: Install the extension and restart Visual Studio Code after installation. 
- If installed successfully, you will see Arena show up on the bottom right corner of your window and the check mark changes to a spinning circle when a completion is being generated, 
- Note, if you are using any other completion provider (e.g. Github Copilot), you must disable them when using Copilot Arena.

Step 2: Copilot Arena currently supports two main feature: read [autocomplete](#autocomplete) and [in-line editing (beta)](#in-line-editing-beta) below to understand how to use each one. Since we show paired responses, the way you use them are slightly different than your standard AI coding tools!

Step 3: This step is optional. If applicable, you can change what data is saved by Copilot Arena by following the instructions in ["Privacy Settings"](#privacy-settings).

Step 4: Create a username by clicking the `Copilot Arena` icon on the sidebar; detailed instructions are also in ["Create an account"](#create-an-account). Your username will be used for a future leaderboard to compare individual preferences.

You are now ready to get coding with Copilot Arena!

## Table of Contents

- [In-line editing](#in-line-editing)
- [Autocomplete](#autocomplete)
- [Personal Leaderboards](#personal-leaderboards)
- [Privacy Settings](#privacy-settings)


## In-line Editing

In-line edits allow you to modify larger portions of your code with a simple prompt.

<img src="https://raw.githubusercontent.com/lm-sys/copilot-arena/89a1155e6b04ab3ff1b4c04c2ddd0b226a1842f7/assets/img/inline1.png" alt="In-line Editing Example" width="850"/>


Here's how to generate in-line edits:
- **Invoke edits:** Highlight any relevant code from your current file and press ```⌘+i``` (Windows: ```Ctrl+i```). Enter in your prompt (e.g., what would you like changed about the highlighted code). When edits are being generated, the check mark changes to a spinning circle.
- **View paired edits:** Two new text files containing diffs between your original file and edited file will pop up. Similar to [autocomplete](#autocomplete), you will see a *pair* of edits.
- **Select between edits:** 
  - Accept left edit: Mac ```⌘+1```, Windows ```Ctrl+1```
  - Accept right edit: Mac ```⌘+2```, Windows ```Ctrl+2```
  - Accept neither (revert): Mac ```⌘+3```, Windows ```Ctrl+3```. For old users, this was originally ```⌘+n```.

Any changes will be automatically applied and the files showing diffs will be closed. You can always undo the change after if you would like.

We welcome feedback on this feature! Please get in touch via Discord or raise an issue on the repo.

## Autocomplete

![Copilot Arena Example](https://raw.githubusercontent.com/lm-sys/copilot-arena/36377762f69231241dcff00817cb8801859612e7/assets/img/example.png)

**Understanding completions:** When a completion is being generated, the check mark changes to a spinning circle. Once a pair of completions appear, you will notice that Copilot Arena adopts a slightly different user interface compared to a typical code completion.

1. Copilot Arena displays two completions, one on top of the other.
2. Copilot Arena repeats the same line prefix to keep the top and bottom outputs as similar as possible.

**Accepting completions:** Press ```Tab``` to accept the top completion and ```Shift-Tab``` to accept the bottom completion. You can also choose to accept neither completion and continue typing.

## Personal Leaderboards

We've observed that individuals have different model preferences! We currently support personal leaderboards for autocomplete. In-line editing leaderboard is coming soon!


![Preference Example](https://raw.githubusercontent.com/lm-sys/copilot-arena/36377762f69231241dcff00817cb8801859612e7/assets/img/model_pref_leaderboard.png)


- When you accept a completion, you can find out which model generated it by checking the status bar. 
- After twenty votes, you will unlock your personal leaderboard, where you can find out which models you prefer and their corresponding elo scores. Based on our experience, everyone will have slightly different model preferences. Stay tuned for a global leaderboard. 
- You can find the personal leaderboard by clicking on the `Copilot Arena` icon on the sidebar.

### Create an account

Soon we will be releasing public leaderboards to compare individual preferences. To enable you to find your position on these leaderboards and help us collect helpful background information, creating an account is as simple as:
- Opening the `Copilot Arena` icon on the sidebar and click `Register`.
- Filling out the requested information.


## Privacy Settings

Your privacy is important to us. Please read carefully to determine which settings are most appropriate for you. 

To generate completions, the code in your current file is sent to our servers and sent to various API providers. This cannot be changed. 

### Data Collection

By default, we collect your code for research purposes. You can opt-out of this. If you are working on code containing sensitive information, we recommend that you opt out of data collection.

- To opt-out of data collection, please change `arena.codePrivacySettings` to `Debug`. We will only log your code for debugging.
- To disable logging entirely, please change `arena.codePrivacySettings` to `Private`. Opting-out means any bugs you encounter will be non-reproducable on our end.

You can find these settings by searching for "arena" in your vscode settings or clicking the gear button of the Copilot Arena extension -> Extension Settings.


### Removing your data

If you would like to have the option in the future for us to delete any of your data, you must create an account on Copilot Arena following instructions described in ["Create an account"](#create-an-account). To remove your data, you can email any of the Copilot Arena maintainers with your username.


### Data Release

Prior to releasing any collected code snippets to enable future research efforts, we will run a PII detector and remove any identified entities to further ensure no personal information is released.

## Code Arena Settings

Code Arena provides several settings. You can find all settings by searching for "arena" in your vscode settings.

| Setting | Description |
|---------|-------------|
| arena.enableTabAutocomplete  | Enable Arena's tab autocomplete feature. |
| arena.enableAutomaticTabAutocomplete | Enables automatic autocomplete suggestions while typing. Turn off to force manual invocations. |
| arena.maxOutputLines | Limit the number of lines of code that can be outputted. Default is 5 lines. |
| arena.displayCompletionIds | Display completion ids (next to the =====). This is used for debugging purposes. |
| arena.codePrivacySettings | Choose your privacy settings for Code Arena:\n- Private: No code is collected\n- Debug: Code is only collected as debug logs\n- Research: Code is uploaded to Code Arena's private servers for research purposes. Your votes are always stored. |
| arena.enableInlineEditHover | Controls whether to show the inline edit hover text when hovering over code or making selections. |
| arena.modelTags | Experimental (WIP) Select the model tags to pick which models you would like to use. Multiple tags can be selected. If no tag is selected, all models are used. |

## Have a question or feedback?

[Check out our Github Repository!](https://github.com/lm-sys/copilot-arena)
