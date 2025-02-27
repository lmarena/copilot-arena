export enum PrivacySetting {
    Private = "Private",
    Debug = "Debug",
    Research = "Research"
}

export enum TaskType {
    Completion = "Completion",
    Edit = "Edit"
}

// Minimizes changes here as it requires changes to the cache.
export interface ArenaCompletionItem {
    completionId: string;
    userId: string;
    timestamp: number;
    prompt: string;
    completion: string;
    model: string;
}

export interface CompletionPairResponse {
    pairId: string;
    completionItems: ArenaCompletionItem[];
}

// Focus on changes to this if tracking new information
export interface CompletionPairRequest {
    pairId: string;
    userId: string;
    prefix: string;
    suffix: string;
    midSpan: boolean;  // Does the completion occur in the middle of a span
    temperature: number;
    maxTokens: number;
    topP: number;
    maxLines: number;
    privacy: PrivacySetting;
    modelTags: string[];
}

export interface ModelSelection {
    model0: string;
    model1: string;
    selectedModel: 0 | 1;
    timestamp: number;
    task?: TaskType;
}

export interface ArenaEditItem {
    editId: string;
    userId: string;
    timestamp: number;
    prompt: string;
    response: string;
    model: string;
    language: string;
    codeToEdit: string;
    userInput: string;
}

export interface EditPairRequest {
    pairId: string;
    userId: string;
    prefix: string;
    suffix: string;
    maxLines: number;
    privacy: PrivacySetting;
    language: string;
    codeToEdit: string;
    userInput: string;
}

export interface EditPairResponse {
    pairId: string;
    responseItems: ArenaEditItem[];
}
