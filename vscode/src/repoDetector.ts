import * as vscode from 'vscode';
import { exec } from 'child_process';
import { promisify } from 'util';
import * as path from 'path';
import * as fs from 'fs';
import { PrivacySetting } from './types';

const execAsync = promisify(exec);

export interface RepoInfo {
    isInRepo: boolean;
    repoUrl?: string;
}

export class RepoDetector {
    static async detect(filePath: string): Promise<RepoInfo> {
        try {
            // Find the git root directory by walking up the directory tree
            const gitRoot = await this.findGitRoot(filePath);
            if (!gitRoot) {
                return { isInRepo: false };
            }

            // Get the remote URL if available
            const remoteUrl = await this.getRemoteUrl(gitRoot);

            return {
                isInRepo: true,
                repoUrl: remoteUrl
            };

        } catch (error) {
            console.error('Error detecting repository:', error);
            return { isInRepo: false };
        }
    }

    private static async findGitRoot(startPath: string): Promise<string | null> {
        let currentPath = startPath;

        while (currentPath !== path.dirname(currentPath)) {
            const gitPath = path.join(currentPath, '.git');

            try {
                const stats = await fs.promises.stat(gitPath);
                if (stats.isDirectory()) {
                    return currentPath;
                }
            } catch { } // Ignore errors and continue searching

            currentPath = path.dirname(currentPath);
        }

        return null;
    }

    private static async getRemoteUrl(repoPath: string): Promise<string | undefined> {
        try {
            const { stdout } = await execAsync('git config --get remote.origin.url', {
                cwd: repoPath
            });
            return stdout.trim();
        } catch {
            return undefined;
        }
    }
}

export async function getRepoInfo(privacy: PrivacySetting): Promise<RepoInfo> {
    if (privacy !== PrivacySetting.Research) {
        return { isInRepo: false };
    }

    const editor: vscode.TextEditor | undefined = vscode.window.activeTextEditor;
    if (editor) {
        const repoInfo = await RepoDetector.detect(editor.document.uri.fsPath);
        return repoInfo;
    }

    return { isInRepo: false };
}
