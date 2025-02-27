const path = require('path');

module.exports = {
  entry: './media/rankingChart.tsx',
  output: {
    path: path.resolve(__dirname, 'media'),
    filename: 'rankingChart.js'
  },
  module: {
    rules: [
      {
        test: /\.tsx?$/,
        use: 'ts-loader',
        exclude: /node_modules/,
      },
    ],
  },
  resolve: {
    extensions: ['.tsx', '.ts', '.js'],
  },
  externals: {
    vscode: 'commonjs vscode' // Exclude vscode module from bundling
  },
};