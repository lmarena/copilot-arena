// The module 'vscode' contains the VS Code extensibility API
// Import the module and reference it with the alias vscode in your code below
import * as vscode from 'vscode';
import { ArenaInlineCompletionProvider as ArenaCompletionProvider } from './inlineCompletionProvider';
import { insertCompletionItem } from './utils';
import { setupStatusBar } from './statusBar';
import * as path from 'path';
import * as os from 'os';
import { CompletionCache } from './cache';
import { ArenaCompletionItem, CompletionPairResponse, EditPairRequest, EditPairResponse, PrivacySetting, TaskType } from './types';
import { uploadArenaPair, uploadArenaSingle } from './api';
import { ModelSelectionViewProvider } from './modelSelectionViewProvider';
import { RankingChartViewProvider } from './rankingChartViewProvider';
import { Disposable } from 'vscode';
import { PromptToDiffHandler } from './promptToDiff';
import { VerticalPerLineDiffManager } from './diff/verticalPerLine/manager';

// This method is called when your extension is activated
// Your extension is activated the very first time the command is executed
export function activate(context: vscode.ExtensionContext) {
	console.log('Congratulations, your extension "copilot-arena" is now active!');
	console.log('sql.js path:', require.resolve('sql.js'));

	// Tab autocomplete
	const config = vscode.workspace.getConfiguration("arena");
	const enabled = config.get<boolean>("enableTabAutocomplete") || false;

	// Ensure that this agreement comes before and command registrations
	const hasAgreedToTos = context.globalState.get<boolean>('hasAgreedToTos');
	if (!hasAgreedToTos) {
		vscode.window.showInformationMessage(
			'You must agree to the following:\n\n' +
			'The service is a research preview. It only provides limited safety measures and may generate offensive content. It must not be used for any illegal, harmful, violent, racist, or sexual purposes. Do not work on code containing any private information. By default, the service collects user development data, including code and votes, and reserves the right to distribute it under a Creative Commons Attribution (CC-BY) or a similar license. You can change your privacy settings in the extension settings. Please see the README for further details.',
			'Agree'
		).then(selection => {
			if (selection === 'Agree') {
				context.globalState.update('hasAgreedToTos', true);
				vscode.commands.executeCommand('workbench.action.reloadWindow');
			} else {
				vscode.window.showInformationMessage('You must agree to the terms to use Copilot Arena.');
				vscode.commands.executeCommand('workbench.action.reloadWindow');
			}
		});
	}

	if (!hasAgreedToTos) {
		return;
	}

	// Check if the warning message has been shown before
	const hasShownCopilotWarning = context.globalState.get<boolean>('hasShownCopilotWarning');

	if (!hasShownCopilotWarning) {
		const copilotExtension = vscode.extensions.getExtension('github.copilot');
		if (copilotExtension) {
			vscode.window.showWarningMessage('GitHub Copilot is installed. Please disable Github Copilot Completions (bottom right) before using Copilot Arena. You do not need to disable Github Copilot Chat. To use Github Copilot again later, disable Copilot Arena and reload your window.');

			// Set the flag to indicate that the warning has been shown
			context.globalState.update('hasShownCopilotWarning', true);
		}
	}

	// Check if the warning message has been shown before
	const hasShownResearchWarning = context.globalState.get<boolean>('hasShownResearchWarning');

	if (!hasShownResearchWarning) {
		const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
		if (privacySetting === PrivacySetting.Research) {
			vscode.window.showWarningMessage('Research mode is enabled and data is being collected. To opt-out of data collection, please change your privacy setting to `Debug` or `Private` in the extension settings. Check out the extension details for more information.');
			// Set the flag to indicate that the warning has been shown
			context.globalState.update('hasShownResearchWarning', true);
		}
	}

	const inlineEditPopUpShown = context.globalState.get('inlineEditPopUpShown', false);

	if (!inlineEditPopUpShown) {
		const keyBinding = process.platform === 'darwin' ? '⌘' : 'Ctrl';
		vscode.window.showInformationMessage(
			`✨ Try out Inline Editing in Copilot Arena! Highlight your code, press ${keyBinding}+i, and enter your prompt. Press ${keyBinding}+1 for left, ${keyBinding}+2 for right, or ${keyBinding}+n neither.`,
			'Cool!'
		);
		context.globalState.update('inlineEditPopUpShown', true);
	}


	const arenaDirPath = path.join(os.homedir(), '.copilot-arena');
	const cache = new CompletionCache(arenaDirPath);
	context.globalState.update('username', null);

	// Register inline completion provider
	const completionProvider = new ArenaCompletionProvider(cache);
	const inlineCompletionProvider = vscode.languages.registerInlineCompletionItemProvider(
		[{ pattern: '**' }],
		completionProvider
	);
	context.subscriptions.push(inlineCompletionProvider);

	const modelSelectionViewProvider = new ModelSelectionViewProvider(context.extensionUri, context);
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(ModelSelectionViewProvider.viewType, modelSelectionViewProvider)
	);

	const rankingChartViewProvider = new RankingChartViewProvider(context.extensionUri, context);
	context.subscriptions.push(
		vscode.window.registerWebviewViewProvider(RankingChartViewProvider.viewType, rankingChartViewProvider)
	);

	// Add this new command registration
	context.subscriptions.push(
		vscode.commands.registerCommand('arena.openExternalUrl', (url: string) => {
			vscode.env.openExternal(vscode.Uri.parse(url));
		})
	);

	context.subscriptions.push(
		vscode.window.onDidChangeTextEditorSelection(() => {
			completionProvider.isPairSuggestionVisible = false;
			completionProvider.isSingleSuggestionVisible = false;
			vscode.commands.executeCommand('hideSuggestWidget');
		})
	);

	let selectFirstDisposable: Disposable | undefined;
	let selectSecondDisposable: Disposable | undefined;

	function registerTabCommands(enabled: boolean) {
		if (enabled) {
			if (!selectFirstDisposable) {
				selectFirstDisposable = vscode.commands.registerCommand('arena.selectFirstInlineCompletion', async () => {
					const config = vscode.workspace.getConfiguration("arena");
					const enabled = config.get("enableTabAutocomplete");

					if (enabled) {
						// This forcefully clears the completions
						completionProvider.isClearingCompletions = true;
						vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
					}

					if (enabled && completionProvider.isPairSuggestionVisible) {
						const completionItem = completionProvider.getFirstInlineCompletion();
						insertCompletionItem(completionItem);

					} else if (enabled && completionProvider.isSingleSuggestionVisible) {
						const completionItem = completionProvider.getSingleInlineCompletion();
						insertCompletionItem(completionItem);
					} else {
						// Check if there are visible suggestions
						vscode.commands.executeCommand('acceptSelectedSuggestion').then(accepted => {
							if (!accepted) {
								// If no suggestions were accepted, insert a tab
								vscode.commands.executeCommand('tab');
							}
						});
					}
					completionProvider.isPairSuggestionVisible = false;
					completionProvider.isSingleSuggestionVisible = false;

					await rankingChartViewProvider.fetchAndUpdateAllData(vscode.env.machineId);
				});
				context.subscriptions.push(selectFirstDisposable);
			}
			if (!selectSecondDisposable) {
				selectSecondDisposable = vscode.commands.registerCommand('arena.selectSecondInlineCompletion', async () => {
					const config = vscode.workspace.getConfiguration("arena");
					const enabled = config.get("enableTabAutocomplete");

					if (enabled) {
						// This forcefully clears the completions
						completionProvider.isClearingCompletions = true;
						vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
					}

					if (enabled && completionProvider.isPairSuggestionVisible) {
						const completionItem = completionProvider.getSecondInlineCompletion();
						insertCompletionItem(completionItem);
					} else {
						// Check if there are visible suggestions
						vscode.commands.executeCommand('acceptSelectedSuggestion').then(accepted => {
							if (!accepted) {
								// If no suggestions were accepted, insert a tab
								vscode.commands.executeCommand('tab');
							}
						});
					}
					completionProvider.isPairSuggestionVisible = false;
					completionProvider.isSingleSuggestionVisible = false;

					await rankingChartViewProvider.fetchAndUpdateAllData(vscode.env.machineId);
				});
				context.subscriptions.push(selectSecondDisposable);
			}

			// Register keybindings programmatically
			vscode.commands.executeCommand('setContext', 'arena.tabAutocompleteEnabled', true);
		} else {
			if (selectFirstDisposable) {
				selectFirstDisposable.dispose();
				selectFirstDisposable = undefined;
			}
			if (selectSecondDisposable) {
				selectSecondDisposable.dispose();
				selectSecondDisposable = undefined;
			}

			// Unregister keybindings programmatically
			vscode.commands.executeCommand('setContext', 'arena.tabAutocompleteEnabled', false);
		}
	}

	// Initial registration based on current setting
	registerTabCommands(enabled);

	// Listen for configuration changes
	const diffManager = new VerticalPerLineDiffManager();

	// Update command registrations
	context.subscriptions.push(
		vscode.commands.registerCommand('arena.acceptFirstLLMResponse', async () => {
			const editPairResponse = await diffManager.acceptLLMResponse(1);
			if (editPairResponse) {
				const model0 = editPairResponse.responseItems[0].model;
				const model1 = editPairResponse.responseItems[1].model;
				vscode.window.setStatusBarMessage(`[${model1}] beats [${model0}]!`, 10000);
				modelSelectionViewProvider.addSelection(model0, model1, 0, TaskType.Edit);
				rankingChartViewProvider.updateSelectionHistory();
			}
		}),
		vscode.commands.registerCommand('arena.acceptSecondLLMResponse', async () => {
			const editPairResponse = await diffManager.acceptLLMResponse(2);
			if (editPairResponse) {
				const model0 = editPairResponse.responseItems[0].model;
				const model1 = editPairResponse.responseItems[1].model;
				vscode.window.setStatusBarMessage(`[${model1}] beats [${model0}]!`, 10000);
				modelSelectionViewProvider.addSelection(model0, model1, 1, TaskType.Edit);
				rankingChartViewProvider.updateSelectionHistory();
			}
		}),
		vscode.commands.registerCommand('arena.rejectAllResponses', () => {
			diffManager.rejectAllResponses();
		})
	);

	// Remove old command registrations for acceptVerticalDiffBlock and rejectVerticalDiffBlock

	const handler = new PromptToDiffHandler(diffManager);
	const disposable = vscode.commands.registerCommand('arena.promptToDiff', () => {
		handler.handlePrompt();
	});
	context.subscriptions.push(disposable);

	context.subscriptions.push(
		vscode.workspace.onDidChangeConfiguration(e => {
			if (e.affectsConfiguration("arena.enableTabAutocomplete")) {
				const newEnabled = vscode.workspace.getConfiguration("arena").get<boolean>("enableTabAutocomplete", false);
				registerTabCommands(newEnabled);
			}
		})
	);

	context.subscriptions.push(
		vscode.commands.registerCommand('arena.toggleTabAutocompleteEnabled', () => {
			const config = vscode.workspace.getConfiguration("arena");
			const enabled = config.get("enableTabAutocomplete");
			config.update(
				"enableTabAutocomplete",
				!enabled,
				vscode.ConfigurationTarget.Global,
			);
			// The onDidChangeConfiguration event will handle updating the commands
		}),
		vscode.commands.registerCommand('arena.finishFirstOutcomeSuccess', (completionPair: CompletionPairResponse, fileType: string) => {
			const config = vscode.workspace.getConfiguration("arena");
			const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
			const model0 = completionPair.completionItems[0].model;
			const model1 = completionPair.completionItems[1].model;
			const selectedModel = 0; // Assuming model 1 is selected, adjust as needed
			cache.removeCompletionById(completionPair.completionItems[0].completionId);
			cache.removeCompletionById(completionPair.completionItems[1].completionId);
			const username = context.globalState.get<string>('username') || '';
			uploadArenaPair(completionPair, selectedModel, privacySetting, fileType, username);
			vscode.window.setStatusBarMessage(`[${model0}] beats [${model1}]!`, 10000);
			modelSelectionViewProvider.addSelection(model0, model1, selectedModel, TaskType.Completion);
			rankingChartViewProvider.updateSelectionHistory(); // Add this line
			if (enabled) {
				// This forcefully clears the completions
				completionProvider.isClearingCompletions = true;
				vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
			}
		}),
		vscode.commands.registerCommand('arena.finishSecondOutcomeSuccess', (completionPair: CompletionPairResponse, fileType: string) => {
			const config = vscode.workspace.getConfiguration("arena");
			const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
			const model0 = completionPair.completionItems[0].model;
			const model1 = completionPair.completionItems[1].model;
			const selectedModel = 1; // Assuming model 1 is selected, adjust as needed
			cache.removeCompletionById(completionPair.completionItems[0].completionId);
			cache.removeCompletionById(completionPair.completionItems[1].completionId);
			const username = context.globalState.get<string>('username') || '';
			uploadArenaPair(completionPair, selectedModel, privacySetting, fileType, username);
			vscode.window.setStatusBarMessage(`[${model1}] beats [${model0}]!`, 10000);
			modelSelectionViewProvider.addSelection(model0, model1, selectedModel, TaskType.Completion);
			rankingChartViewProvider.updateSelectionHistory(); // Add this line
			if (enabled) {
				// This forcefully clears the completions
				completionProvider.isClearingCompletions = true;
				vscode.commands.executeCommand("editor.action.inlineSuggest.trigger");
			}
		}),
		vscode.commands.registerCommand('arena.finishSingleOutcomeSuccess', (completionItem: ArenaCompletionItem) => {
			const config = vscode.workspace.getConfiguration("arena");
			const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
			cache.removeCompletionById(completionItem.completionId);
			uploadArenaSingle(completionItem, privacySetting);
		}),
		vscode.commands.registerCommand('arena.clearCompletionsCache', () => {
			cache.clearCache();
		}),
		vscode.commands.registerCommand("arena.showOptions", () => {
			const options = [
				"Toggle Tab Autocomplete",
				"Join Discord Community",
				"Submit Bug Report"
			];

			vscode.window.showQuickPick(options).then(selection => {
				switch (selection) {
					case "Toggle Tab Autocomplete":
						vscode.commands.executeCommand("arena.toggleTabAutocompleteEnabled");
						break;
					case "Join Discord Community":
						vscode.env.openExternal(vscode.Uri.parse("https://discord.gg/ftfqdMNh3B"));
						break;
					case "Submit Bug Report":
						vscode.env.openExternal(vscode.Uri.parse("https://github.com/lmarena/copilot-arena/issues/new"));
						break;
				}
			});
		})
	);

	setupStatusBar(enabled);

	const isMac = process.platform === 'darwin';
	const keyBinding = isMac ? '⌘+I' : 'Ctrl+I';
	const hoverText = `✨ Press **${keyBinding}** to activate Copilot Arena Inline Editing! ✨`;
	// Add hover provider
	const enableHover = config.get<boolean>("enableInlineEditHover") ?? true;
	if (enableHover) {
		const hoverProvider = vscode.languages.registerHoverProvider({ pattern: '**' }, {
			provideHover(document, position, token) {
				if (!enableHover) {
					return null;
				}
				return new vscode.Hover(hoverText);
			}
		});
		context.subscriptions.push(hoverProvider);
	}

	// Add selection change handler
	context.subscriptions.push(
		vscode.window.onDidChangeTextEditorSelection(event => {
			const selection = event.textEditor.selection;
			if (!selection.isEmpty) {
				vscode.window.setStatusBarMessage(hoverText);
			}
		})
	);
}

// This method is called when your extension is deactivated
export function deactivate() { }