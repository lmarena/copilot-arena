(function () {
    const vscode = acquireVsCodeApi();
    const selectionList = document.getElementById('selectionList');
    const clearButton = document.getElementById('clearButton');
    const sortButton = document.getElementById('sortButton');
    let selections = [];
    let sortAscending = false;

    clearButton.addEventListener('click', () => {
        vscode.postMessage({ type: 'clearHistory' });
    });

    sortButton.addEventListener('click', () => {
        sortAscending = !sortAscending;
        sortButton.textContent = `Sort: Recent ${sortAscending ? '↑' : '↓'}`;
        vscode.postMessage({ type: 'toggleSort', sortAscending });
    });

    function addSelectionToList(selection) {
        const li = document.createElement('li');
        const timestamp = new Date(selection.timestamp).toLocaleString();
        li.innerHTML = `
            <div>
                <span class="model ${selection.selectedModel === 0 ? 'selected' : ''}">${selection.model0}</span>
                <span class="model ${selection.selectedModel === 1 ? 'selected' : ''}">${selection.model1}</span>
                <div class="timestamp">${timestamp}</div>
            </div>
        `;
        return li;
    }

    function renderSelections() {
        selectionList.innerHTML = '';
        const sortedSelections = [...selections].sort((a, b) =>
            sortAscending ? a.timestamp - b.timestamp : b.timestamp - a.timestamp
        );
        sortedSelections.forEach(selection => {
            selectionList.appendChild(addSelectionToList(selection));
        });
    }

    window.addEventListener('message', event => {
        const message = event.data;
        switch (message.type) {
            case 'initializeSelections':
                selections = message.selections;
                sortAscending = message.sortAscending;
                sortButton.textContent = `Sort: Recent ${sortAscending ? '↑' : '↓'}`;
                renderSelections();
                break;
            case 'clearSelections':
                selections = [];
                renderSelections();
                break;
            case 'addSelection':
                selections.unshift(message.selection);
                renderSelections();
                break;
        }
    });
})();