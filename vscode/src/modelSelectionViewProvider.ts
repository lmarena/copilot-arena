import * as vscode from 'vscode';
import { ModelSelection, TaskType } from './types';


export class ModelSelectionViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'modelSelectionView';
    private _view?: vscode.WebviewView;
    private _selections: ModelSelection[] = [];
    private _sortAscending: boolean = false;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _context: vscode.ExtensionContext
    ) {
        // Load saved selections when the provider is created
        this._selections = (() => {
            return this._context.globalState.get<ModelSelection[]>('modelSelections', []);
        })();
        this._sortAscending = (() => {
            return this._context.globalState.get<boolean>('sortAscending', false);
        })();
    }

    public resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [this._extensionUri]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.onDidChangeVisibility(() => {
            if (webviewView.visible) {
                this.updateWebview();
            }
        });

        webviewView.webview.onDidReceiveMessage(data => {
            if (data.type === 'clearHistory') {
                this._selections = [];
                this._context.globalState.update('modelSelections', this._selections);
                this.updateWebview();
            } else if (data.type === 'toggleSort') {
                this._sortAscending = data.sortAscending;
                this._context.globalState.update('sortAscending', this._sortAscending);
                this.updateWebview();
            }
        });

        // Initial update
        this.updateWebview();
    }

    private updateWebview() {
        if (this._view) {
            this._view.webview.postMessage({
                type: 'initializeSelections',
                selections: this._selections,
                sortAscending: this._sortAscending
            });
        }
    }

    public addSelection(model0: string, model1: string, selectedModel: 0 | 1, task: TaskType) {
        const newSelection: ModelSelection = {
            model0,
            model1,
            selectedModel,
            timestamp: Date.now(),
            task: task
        };
        this._selections.unshift(newSelection);
        this._context.globalState.update('modelSelections', this._selections);
        this.updateWebview();
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'main.js'));

        return `<!DOCTYPE html>
                <html lang="en">
                <head>
                    <meta charset="UTF-8">
                    <meta name="viewport" content="width=device-width, initial-scale=1.0">
                    <title>Model Selection History</title>
                    <style>
                        body {
                            font-family: var(--vscode-font-family);
                            font-size: var(--vscode-font-size);
                            color: var(--vscode-foreground);
                            padding: 0 20px;
                            line-height: 1.4;
                        }
                        h1 {
                            font-size: 1.2em;
                            font-weight: normal;
                            margin-bottom: 1em;
                            border-bottom: 1px solid var(--vscode-panel-border);
                            padding-bottom: 0.5em;
                        }
                        #clearButton, #sortButton {
                            background-color: var(--vscode-button-background);
                            color: var(--vscode-button-foreground);
                            border: none;
                            padding: 6px 12px;
                            cursor: pointer;
                            font-size: 0.9em;
                            border-radius: 2px;
                        }
                        #clearButton:hover, #sortButton:hover {
                            background-color: var(--vscode-button-hoverBackground);
                        }
                        #selectionList {
                            list-style-type: none;
                            padding: 0;
                            margin-top: 1em;
                        }
                        #selectionList li {
                            background-color: var(--vscode-editor-background);
                            border: 1px solid var(--vscode-panel-border);
                            margin-bottom: 8px;
                            padding: 8px 12px;
                            border-radius: 4px;
                            display: flex;
                            justify-content: space-between;
                            align-items: center;
                        }
                        .model {
                            display: inline-block;
                            padding: 4px 8px;
                            border-radius: 2px;
                            font-size: 0.9em;
                        }
                        .selected {
                            background-color: var(--vscode-editor-selectionBackground);
                            color: var(--vscode-editor-selectionForeground);
                            font-weight: bold;
                        }
                        .timestamp {
                            font-size: 0.8em;
                            color: var(--vscode-descriptionForeground);
                            margin-top: 4px;
                        }
                    </style>
                </head>
                <body>
                    <h1>Model Selection History</h1>
                    <button id="clearButton">Clear Local History</button>
                    <button id="sortButton">Sort: Recent ↓</button>
                    <ul id="selectionList"></ul>
                    <script src="${scriptUri}"></script>
                </body>
                </html>`;
    }
}