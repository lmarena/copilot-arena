import { setupStatusBar, stopStatusBarLoading } from './statusBar';
import { CompletionPairResponse, CompletionPairRequest, ArenaCompletionItem, PrivacySetting, EditPairRequest, EditPairResponse, ArenaEditItem } from "./types";
import { version } from '../package.json';
import { RepoInfo, getRepoInfo } from './repoDetector';
import * as vscode from 'vscode';

const serverUrl = vscode.workspace.getConfiguration('arena').get<string>('serverUrl', 'https://code-arena.fly.dev');

interface CompletionItemData extends ArenaCompletionItem {
    pairIndex: number;
    version: string;
    privacy: PrivacySetting
    pairCompletionId?: string;
    fileType: string;
    username?: string;
}

interface CompletionPairData {
    pairId: string;
    userId: string;
    completionItems: CompletionItemData[];
    acceptedIndex: number;
    version: string;
    privacy: PrivacySetting;
    fileType: string;
    repoUrl?: string;
}

interface EditItemData extends ArenaEditItem {
    pairIndex: number;
    version: string;
    privacy: PrivacySetting;
    fileType: string;
    username?: string;
}

interface EditPairData {
    pairId: string;
    userId: string;
    responseItems: EditItemData[];
    acceptedIndex: number;
    version: string;
    privacy: PrivacySetting;
    fileType: string;
    repoUrl?: string;
}

interface CompletionSingleData extends ArenaCompletionItem {
    version: string;
}

export interface ModelScore {
    model: string;
    elo: number;
    votes: number;
}


export async function fetchCompletionPair(completionPairRequest: CompletionPairRequest, signal: AbortSignal): Promise<CompletionPairResponse | undefined> {
    try {
        // Abort early if possible
        if (!signal || signal.aborted) {
            return undefined;
        }

        setupStatusBar(true, true);
        const response = await fetch(`${serverUrl}/create_pair`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(completionPairRequest),
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        return await response.json() as CompletionPairResponse;
    } catch (error) {
        console.error('Error fetching completion pair:', error);
        throw error;
    } finally {
        stopStatusBarLoading();
    }
}

export async function uploadArenaCompletion(arenaCompletionItem: ArenaCompletionItem, pairIndex: number, pairCompletionId: string, privacy: PrivacySetting, fileType: string): Promise<void> {
    // Make the HTTP PUT request
    try {
        const completionItemData: CompletionItemData = {
            ...arenaCompletionItem,
            pairIndex,
            version,
            pairCompletionId,
            privacy,
            fileType,
        };
        const response = await fetch(`${serverUrl}/add_completion`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(completionItemData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        console.log("Completion data successfully uploaded.");
    } catch (error) {
        console.error("Failed to upload completion:", error);
    }
}


export async function uploadArenaPair(completionPair: CompletionPairResponse, acceptedIndex: number, privacy: PrivacySetting, fileType: string, username: string): Promise<void> {
    try {
        const repoInfo: RepoInfo = await getRepoInfo(privacy);
        const repoUrl: string | undefined = repoInfo.repoUrl;

        const completionItemData1: CompletionItemData = {
            ...completionPair.completionItems[0],
            pairIndex: 0,
            version: version,
            privacy,
            fileType,
            username,
        };
        const completionItemData2: CompletionItemData = {
            ...completionPair.completionItems[1],
            pairIndex: 1,
            version: version,
            privacy,
            fileType,
            username
        };
        const completionPairData: CompletionPairData = {
            pairId: completionPair.pairId,
            userId: completionPair.completionItems[0].userId,
            completionItems: [completionItemData1, completionItemData2],
            acceptedIndex,
            version,
            privacy,
            fileType,
            repoUrl
        };
        const response = await fetch(`${serverUrl}/add_completion_outcome`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(completionPairData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        console.log("Pair Outcome data successfully uploaded.");
    } catch (error) {
        console.error("Failed to upload outcomes:", error);
    }
}

export async function uploadArenaSingle(arenaCompletionItem: ArenaCompletionItem, privacy: PrivacySetting): Promise<void> {
    try {
        const CompletionSingleData: CompletionSingleData = {
            ...arenaCompletionItem,
            version
        };
        const response = await fetch(`${serverUrl}/add_single_outcome`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(CompletionSingleData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        console.log("Pair Outcome data successfully uploaded.");
    } catch (error) {
        console.error("Failed to upload outcomes:", error);
    }
}

export async function getUserScores(userId: string): Promise<Array<Array<ModelScore>>> {
    try {
        const response = await fetch(`${serverUrl}/user_scores`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ "userId": userId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Personal ranking data successfully fetched.");
        return data;
    } catch (error) {
        console.error("Failed to fetch personal ranking:", error);
        throw error;
    }
}

export async function getUserVoteCount(userId: string): Promise<number> {
    try {
        const response = await fetch(`${serverUrl}/user_vote_count`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ "userId": userId }),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        return data['voteCount'];
    } catch (error) {
        console.error("Failed to fetch user vote count:", error);
        throw error;
    }
}

export async function registerUser(userId: string, username: string, password?: string, metadata?: any): Promise<void> {
    try {
        const response = await fetch(`${serverUrl}/users`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ userId, username, password, metadata }),
        });

        if (response.status === 409) {
            throw new Error('Username already taken');
        }

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        console.log("Username successfully updated.");
    } catch (error) {
        console.error("Failed to update username:", error);
        throw error;
    }
}

export async function login(userId: string, username: string, password?: string): Promise<any> {
    try {
        const response = await fetch(`${serverUrl}/users/authenticate`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ userId, username, password }),
        });

        if (response.status === 401) {
            throw new Error('Incorrect username or password');
        }

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        const data = await response.json();
        console.log("Login successful.");
        return data;
    } catch (error) {
        console.error("Failed to login:", error);
        throw error;
    }
}

export async function fetchEditPair(editPairRequest: EditPairRequest, signal: AbortSignal): Promise<EditPairResponse | undefined> {
    try {
        // Abort early if possible
        if (!signal || signal.aborted) {
            return undefined;
        }

        setupStatusBar(true, true);
        const response = await fetch(`${serverUrl}/create_edit_pair`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(editPairRequest),
        });

        if (!response.ok) {
            throw new Error(`API request failed with status ${response.status}`);
        }

        return await response.json() as EditPairResponse;
    } catch (error) {
        console.error('Error fetching edit pair:', error);
        throw error;
    } finally {
        stopStatusBarLoading();
    }
}


export async function uploadEditPair(editPair: EditPairResponse, acceptedIndex: number, privacy: PrivacySetting, fileType: string, username?: string): Promise<void> {
    try {
        const repoInfo: RepoInfo = await getRepoInfo(privacy);
        const repoUrl: string | undefined = repoInfo.repoUrl;
        const editItemData1: EditItemData = {
            ...editPair.responseItems[0],
            pairIndex: 0,
            version: version,
            privacy,
            fileType,
            username
        };
        const editItemData2: EditItemData = {
            ...editPair.responseItems[1],
            pairIndex: 1,
            version: version,
            privacy,
            fileType,
            username
        };
        const editPairData: EditPairData = {
            pairId: editPair.pairId,
            userId: editPair.responseItems[0].userId,
            responseItems: [editItemData1, editItemData2],
            acceptedIndex,
            version,
            privacy,
            fileType,
            repoUrl
        };
        const response = await fetch(`${serverUrl}/add_edit_outcome`, {
            method: 'PUT',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(editPairData),
        });

        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }

        console.log("Edit Pair Outcome data successfully uploaded.");
    } catch (error) {
        console.error("Failed to upload edit outcomes:", error);
    }
}
