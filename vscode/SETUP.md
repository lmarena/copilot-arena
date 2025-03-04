# Copilot Arena - Setup Guide

This document outlines the setup and development processes for the Copilot Arena VSCode extension.

## Prerequisites

- [Node.js](https://nodejs.org/) (latest LTS version recommended)
- [Visual Studio Code](https://code.visualstudio.com/)
- [Git](https://git-scm.com/)

## Initial Setup

1. Clone the repository:
   ```
   git clone https://github.com/lm-sys/copilot-arena.git
   cd copilot-arena/vscode
   ```

2. Install dependencies:
   ```
   npm install
   ```

## Development Workflow

### Building the Extension

For development builds:
```
npm run compile-dev
```

For production builds:
```
npm run compile
```

The compiled extension will be generated in the `dist` directory.

### Watch Mode

To automatically rebuild the extension when code changes:
```
npm run watch
```

This runs multiple watch processes in parallel:
- TypeScript compiler in watch mode
- esbuild in watch mode
- webpack in watch mode

### Type Checking

To check TypeScript types without emitting any files:
```
npm run check-types
```

### Linting

To lint the codebase:
```
npm run lint
```

## Testing

### Running Tests

To run tests:
```
npm test
```

This will:
1. Compile the test files
2. Compile the extension
3. Run linting
4. Execute the tests

### Running Tests in Watch Mode

During development, you can use:
```
npm run watch-tests
```

This will recompile test files whenever they change.

## Packaging and Publishing

### Packaging the Extension

To package the extension for production:
```
npm run package
```

This command:
1. Checks TypeScript types
2. Runs linting
3. Bundles the extension with esbuild in production mode
4. Bundles the web components with webpack in production mode

### Publishing the Extension

To publish the extension to the Visual Studio Code Marketplace:

1. Make sure you have the [vsce](https://github.com/microsoft/vscode-vsce) tool installed:
   ```
   npm install -g @vscode/vsce
   ```

2. Package the extension:
   ```
   vsce package
   ```
   This creates a `.vsix` file.

3. Publish the extension:
   ```
   vsce publish
   ```
   You'll need the appropriate credentials for the Visual Studio Code Marketplace.

## Project Structure

- `src/`: Main TypeScript source code
  - `extension.ts`: Entry point of the extension
  - `api.ts`: API functionality
  - `diff/`: Code for handling diffs
  - `test/`: Test files
- `media/`: Web components for VSCode views
- `dist/`: Generated extension code
- `out/`: Compiled test files

## Rebuilding After Dependency Changes

If you update dependencies in `package.json`, follow these steps:

1. Install the updated dependencies:
   ```
   npm install
   ```

2. Rebuild the project:
   ```
   npm run compile
   ```

## Troubleshooting

If you encounter build issues:

1. Clean the project:
   ```
   rm -rf dist out
   ```

2. Reinstall dependencies:
   ```
   npm ci
   ```

3. Rebuild the project:
   ```
   npm run compile
   ```

## Additional Notes

- The extension uses esbuild for the main extension code and webpack for web components.
- When modifying the web components in the `media/` directory, the webpack configuration handles building those files.
- SQL.js WASM files are automatically copied to the dist folder during build by a custom esbuild plugin.