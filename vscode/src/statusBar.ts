import * as vscode from "vscode";

// Wayne. This is the status bar. Very cool!
const statusBarItemText = (enabled: boolean | undefined) =>
  enabled ? "$(check) Arena" : "$(circle-slash) Arena";

const statusBarItemTooltip = (enabled: boolean | undefined) =>
  enabled ? "Arena tab autocomplete is enabled" : "Click to enable tab autocomplete";

let statusBarItem: vscode.StatusBarItem | undefined = undefined;
let statusBarFalseTimeout: NodeJS.Timeout | undefined = undefined;

export function stopStatusBarLoading() {
  statusBarFalseTimeout = setTimeout(() => {
    setupStatusBar(true, false);
  }, 100);
}

export function setupStatusBar(
  enabled: boolean | undefined,
  loading?: boolean,
) {
  if (loading !== false) {
    clearTimeout(statusBarFalseTimeout);
    statusBarFalseTimeout = undefined;
  }

  // If statusBarItem hasn't been defined yet, create it
  if (!statusBarItem) {
    statusBarItem = vscode.window.createStatusBarItem(
      vscode.StatusBarAlignment.Right,
    );
  }

  statusBarItem.text = loading
    ? "$(loading~spin) Arena"
    : statusBarItemText(enabled);
  statusBarItem.tooltip = "Click for Arena options";
  statusBarItem.command = "arena.showOptions";

  statusBarItem.show();

  vscode.workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration("arena")) {
      const config = vscode.workspace.getConfiguration("arena");
      const enabled = config.get<boolean>("enableTabAutocomplete");
      setupStatusBar(enabled);
    }
  });
}