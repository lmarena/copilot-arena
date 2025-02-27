import * as vscode from 'vscode';
import { ModelSelection } from './types';
import { getUserScores, getUserVoteCount, ModelScore, registerUser, login } from './api';
import { debugLog } from './utils';


interface ModelRank {
    model: string;
    rank: number;
}

export class RankingChartViewProvider implements vscode.WebviewViewProvider {
    public static readonly viewType = 'rankingChartView';
    private _view?: vscode.WebviewView;
    private _userVoteCount: number = 0;
    private _chartData: Array<Array<ModelRank>> = [];
    private _mostRecentScores: Array<ModelScore> = [];
    private _username: string | null = null;

    constructor(
        private readonly _extensionUri: vscode.Uri,
        private readonly _context: vscode.ExtensionContext
    ) {
        // Initialize the username from the global state
        this._username = this._context.globalState.get('username', null);
    }

    public async resolveWebviewView(
        webviewView: vscode.WebviewView,
        context: vscode.WebviewViewResolveContext,
        _token: vscode.CancellationToken,
    ) {
        this._view = webviewView;

        webviewView.webview.options = {
            enableScripts: true,
            localResourceRoots: [
                this._extensionUri
            ]
        };

        webviewView.webview.html = this._getHtmlForWebview(webviewView.webview);

        webviewView.webview.onDidReceiveMessage(async message => {
            debugLog('Received message from webview:', message);
            switch (message.type) {
                case 'webviewReady':
                    await this.triggerWebviewReady();
                    break;
                case 'openExternalUrl':
                    vscode.env.openExternal(vscode.Uri.parse(message.url));
                    break;
                case 'toggleSort':
                    this._context.globalState.update('sortAscending', message.sortAscending);
                    this.updateSelectionHistory();
                    break;
                case 'getMostRecentUserScores':
                    break;
                case 'register':
                    try {
                        await registerUser(vscode.env.machineId, message.username, message.password, message.metadata);
                        this._username = message.username;
                        webviewView.webview.postMessage({ type: 'loginSuccess', username: message.username });
                        // Set username globally as a variable
                        this._context.globalState.update('username', message.username);
                        vscode.window.showInformationMessage('Registration successful!');
                    } catch (error) {
                        vscode.window.showErrorMessage('Registration failed. Please try a different username.');
                    }
                    break;
                case 'updateLeaderboard':
                    await this.fetchAndUpdateAllData(vscode.env.machineId);
                    break;
            }
        });
    }

    public updateSelectionHistory() {
        if (this._view) {
            const selectionHistory = this._context.globalState.get<ModelSelection[]>('modelSelections', []);
            const sortAscending = this._context.globalState.get<boolean>('sortAscending', false);
            debugLog('Sending updateSelectionHistory message with data:', selectionHistory);
            this._view.webview.postMessage({
                type: 'initializeSelections',
                selections: selectionHistory,
                sortAscending: sortAscending
            });
        } else {
            debugLog('View is not available');
        }
    }

    private _getHtmlForWebview(webview: vscode.Webview) {
        const scriptUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'media', 'rankingChart.js'));
        const reactUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'react', 'umd', 'react.production.min.js'));
        const reactDomUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'react-dom', 'umd', 'react-dom.production.min.js'));
        const rechartsUri = webview.asWebviewUri(vscode.Uri.joinPath(this._extensionUri, 'node_modules', 'recharts', 'umd', 'Recharts.js'));

        return `<!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource}; script-src ${webview.cspSource};">
                <title>Model Ranking Chart</title>
            </head>
            <body>
                <div id="root"></div>
                <script src="${reactUri}"></script>
                <script src="${reactDomUri}"></script>
                <script src="${rechartsUri}"></script>
                <script src="${scriptUri}"></script>
            </body>
            </html>`;
    }

    public updateTimeChart(data: any) {
        if (this._view) {
            debugLog('Sending updateTimeChart message with data:', data);
            this._view.webview.postMessage({ type: 'updateChart', data });
        } else {
            debugLog('View is not available');
        }
    }

    public updateUserScores(data: any) {
        if (this._view) {
            debugLog('Sending updateUserScores message with data:', data);
            this._view.webview.postMessage({ type: 'updateUserScores', data });
        } else {
            debugLog('View is not available');
        }
    }

    public updateUserVoteCount(data: number) {
        if (this._view) {
            debugLog('Sending updateUserVoteCount message with data:', data);
            this._view.webview.postMessage({ type: 'updateUserVoteCount', data });
        } else {
            debugLog('View is not available');
        }
    }

    public async getUserRanking(userScores: Array<Array<ModelScore>>): Promise<Array<Array<ModelRank>>> {
        try {
            // Convert scores to rankings
            const rankings = userScores.map(scoreSet => {
                // Sort the scores in descending order
                const sortedScores = scoreSet.sort((a, b) => b.elo - a.elo);

                // Assign ranks
                return sortedScores.map((score, index) => ({
                    model: score.model,
                    rank: index + 1
                }));
            });

            return rankings;
        } catch (error) {
            console.error('Error fetching user rankings:', error);
            throw error;
        }
    }

    public async getUserVoteCount(userId: string): Promise<number> {
        try {
            const userVotecount = await getUserVoteCount(userId);

            return userVotecount;
        } catch (error) {
            console.error('Error fetching user vote count:', error);
            throw error;
        }
    }

    public getTestData(): Array<Array<{ model: string, rank: number }>> {
        const models = ['Model A', 'Model B', 'Model C', 'Model D', 'Model E', 'Model F', 'Model G'];
        const dataPoints = 10;
        const data: Array<Array<{ model: string, rank: number }>> = [];
        let currentRankings = models.map((model, index) => ({ model, rank: index + 1 }));

        for (let i = 0; i < dataPoints; i++) {
            data.push(this.deepCopy(currentRankings));
            currentRankings = this.shiftRankings(currentRankings);
        }

        return data;
    }

    private shiftRankings(rankings: { model: string, rank: number }[]): { model: string, rank: number }[] {
        const newRankings = [...rankings];

        // Randomly select a model to shift
        const indexToShift = Math.floor(Math.random() * rankings.length);
        const currentRank = newRankings[indexToShift].rank;

        // Determine if the rank should increase or decrease
        const shiftDirection = Math.random() < 0.5 ? -1 : 1;
        const newRank = Math.max(1, Math.min(rankings.length, currentRank + shiftDirection));

        // If the rank didn't change, return the original rankings
        if (newRank === currentRank) {
            return rankings;
        }

        // Adjust the ranks of other models
        newRankings.forEach(ranking => {
            if (shiftDirection < 0 && ranking.rank < currentRank && ranking.rank >= newRank) {
                ranking.rank++;
            } else if (shiftDirection > 0 && ranking.rank > currentRank && ranking.rank <= newRank) {
                ranking.rank--;
            }
        });

        // Set the new rank for the shifted model
        newRankings[indexToShift].rank = newRank;

        return newRankings;
    }

    private deepCopy<T>(obj: T): T {
        return JSON.parse(JSON.stringify(obj));
    }

    // Add this function to get the most recent user scores
    public async getMostRecentUserScores(userScores: Array<Array<ModelScore>>): Promise<Array<ModelScore>> {
        try {
            if (userScores.length === 0) {
                return [];
            }
            return userScores[userScores.length - 1]; // Return the most recent scores
        } catch (error) {
            console.error('Error fetching most recent user scores:', error);
            throw error;
        }
    }

    public async triggerWebviewReady() {
        if (this._view) {
            debugLog('Triggering webview ready');
            const userId = vscode.env.machineId;
            // If username doesn't exist, try to authenticate
            if (!this._username) {
                await this.tryAuthenticate(userId);
            }
            // Send the initial login state to the webview
            if (this._username) {
                this._view.webview.postMessage({ type: 'loginSuccess', username: this._username });
            }

            // Use stored data initially
            this.updateUserVoteCount(this._userVoteCount);
            this.updateTimeChart(this._chartData);
            this.updateUserScores(this._mostRecentScores);

            this.updateSelectionHistory();

            // Fetch the latest data asynchronously
            await this.fetchAndUpdateAllData(userId);
        }
    }

    private async tryAuthenticate(userId: string) {
        try {
            const response = await login(userId, '');
            if (response && response.user && response.user.username) {
                this._username = response.user.username;
                this._context.globalState.update('username', this._username);
                debugLog('Authentication successful, username:', this._username);
            }
        } catch (error) {
            debugLog('Authentication failed. Username likely does not exist:', error);
        }
    }

    public async fetchAndUpdateAllData(userId: string) {
        try {
            const [userScores, userVoteCount] = await Promise.all([
                getUserScores(userId),
                getUserVoteCount(userId)
            ]);

            this._userVoteCount = userVoteCount;
            this.updateUserVoteCount(userVoteCount);

            const rankingData = await this.getUserRanking(userScores);
            this._chartData = rankingData;
            this.updateTimeChart(rankingData);

            const mostRecentScores = await this.getMostRecentUserScores(userScores);
            this._mostRecentScores = mostRecentScores;
            this.updateUserScores(mostRecentScores);
        } catch (error) {
            console.error('Error fetching user data:', error);
        }
    }
}
