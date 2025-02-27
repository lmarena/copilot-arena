import * as vscode from 'vscode';
import { ArenaInlineCompletionProvider } from './inlineCompletionProvider';

const DEBUG_MODE = false;

export function insertCompletionItem(completionItem: vscode.InlineCompletionItem | undefined) {
    if (!completionItem) {
        vscode.window.showErrorMessage('No completion item available');
        return;
    }

    const editor = vscode.window.activeTextEditor;
    if (!editor) {
        vscode.window.showErrorMessage('No active editor');
        return;
    }

    const insertText = typeof completionItem.insertText === 'string' ?
        completionItem.insertText :
        completionItem.insertText.value;

    const range = completionItem.range ?? editor.selection;

    editor.edit((editBuilder) => {
        editBuilder.replace(range, insertText);
    }).then(success => {
        if (success && completionItem.command) {
            vscode.commands.executeCommand(completionItem.command.command, ...completionItem.command.arguments || []);
        }
    });

    vscode.commands.executeCommand('editor.action.inlineSuggest.trigger', []);
}

export function getFirstCommand(completionProvider: ArenaInlineCompletionProvider) {
    return vscode.commands.registerCommand('arena.selectFirstInlineCompletion', () => {
        const config = vscode.workspace.getConfiguration("arena");
        const enabled = config.get("enableTabAutocomplete");

        if (enabled && completionProvider.isPairSuggestionVisible) {
            const completionItem = completionProvider.getFirstInlineCompletion();
            insertCompletionItem(completionItem);
        } else if (enabled && completionProvider.isSingleSuggestionVisible) {
            const completionItem = completionProvider.getSingleInlineCompletion();
            insertCompletionItem(completionItem);
        } else {
            vscode.commands.executeCommand('tab');
        }
        completionProvider.isPairSuggestionVisible = false;
        completionProvider.isSingleSuggestionVisible = false;
    });
}

export function getSecondCommand(completionProvider: ArenaInlineCompletionProvider) {
    vscode.commands.registerCommand('arena.selectSecondInlineCompletion', () => {
        const config = vscode.workspace.getConfiguration("arena");
        const enabled = config.get("enableTabAutocomplete");

        if (enabled && completionProvider.isPairSuggestionVisible) {
            const completionItem = completionProvider.getSecondInlineCompletion();
            insertCompletionItem(completionItem);
        } else {
            vscode.commands.executeCommand('shift+tab');
        }
        completionProvider.isPairSuggestionVisible = false;
        completionProvider.isSingleSuggestionVisible = false;
    });
}

export function isMidSpan(prefix: string) {
    return !/\n\s*$/.test(prefix);
}

export function debugLog(...args: any[]) {
    if (DEBUG_MODE) {
        console.log(...args);
    }
}