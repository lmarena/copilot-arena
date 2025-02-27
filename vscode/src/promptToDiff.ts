import * as vscode from 'vscode';
import { VerticalPerLineDiffManager } from './diff/verticalPerLine/manager';
import { streamDiffLines } from './diff/diff';
import { fetchEditPair } from './api';
import { EditPairRequest, EditPairResponse, PrivacySetting } from './types';
import { randomUUID } from 'crypto';

export class PromptToDiffHandler {
    private diffManager: VerticalPerLineDiffManager;

    constructor(diffManager: VerticalPerLineDiffManager) {
        this.diffManager = diffManager;
    }

    public async handlePrompt() {
        const config = vscode.workspace.getConfiguration("arena");
        const privacySetting = config.get<PrivacySetting>("codePrivacySettings") || PrivacySetting.Private;
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            vscode.window.showErrorMessage('No active editor');
            return;
        }
        const prompt = await vscode.window.showInputBox({
            prompt: 'Enter your prompt for code generation',
            placeHolder: 'e.g., Add a function to calculate factorial',
        });
        if (!prompt) {
            return; // User cancelled the input
        }

        const document = editor.document;
        const selection = editor.selection;
        const startLine = selection.start.line;
        const endLine = selection.end.line;

        const prefix = document.getText(new vscode.Range(0, 0, startLine, 0));
        const highlighted = document.getText(selection);
        const suffix = document.getText(new vscode.Range(endLine, document.lineAt(endLine).text.length, document.lineCount, 0));

        const editPairRequest: EditPairRequest = {
            pairId: randomUUID(),
            userId: vscode.env.machineId,
            prefix,
            suffix,
            maxLines: 50,
            privacy: privacySetting,
            language: document.languageId,
            codeToEdit: highlighted,
            userInput: prompt
        };

        try {
            const controller = new AbortController();
            const editPairResponse = await fetchEditPair(editPairRequest, controller.signal);

            if (editPairResponse) {
                await this.diffManager.streamSideBySideDiff(
                    document.uri,
                    new vscode.Range(startLine, 0, endLine, document.lineAt(endLine).text.length),
                    editPairResponse.responseItems[0].response,
                    editPairResponse.responseItems[1].response,
                    editPairResponse
                );
            }
        } catch (error) {
            console.error('Error fetching edit pair:', error);
            vscode.window.showErrorMessage('Failed to generate code edits. Please try again.');
        }
    }
}
