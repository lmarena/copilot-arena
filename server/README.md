# Copilot Arena

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> Copilot Arena server which handles requests to various model providers for AI code completion and editing.

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Development](#development) • [Contributing](#contributing) • [License](#license)

Download the Copilot Arena extension from the [Visual Studio Code Store](https://marketplace.visualstudio.com/items?itemName=copilot-arena.copilot-arena). 
For instructions on how to use Copilot Arena, check out the main [Copilot Arena Github Page](https://github.com/lmarena/copilot-arena/tree/main).

## Features

- 🚀 Handles multiple model providers
- 🔄 Code completion capabilities
- ✏️  Inline code editing
- 🛠️ Easy-to-use API interface
- 📝 Customizable prompt templates

## Installation

```bash
# Clone the repository
git clone https://github.com/waynchi/copilot-arena-server.git

# Navigate to the directory
cd copilot-arena-server

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Code Completions

Prompt templates for code completions are located in:
```
templates/chat_psm_overlap.yaml
```

### Inline Edits

Editing templates can be found in:
```
templates/edit/chat_edit.yaml
```

## Development

### Local Setup

For detailed development instructions, please refer to our [Development Guide](config/DEV_README.md).

### Prerequisites

- Python 3.11+
- pip
- conda (recommended)

### Running Tests

```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
pytest
```

## Contributing

We welcome contributions! Here's how you can help:

1. 🍴 Fork the repository
2. 🌿 Create your feature branch (`git checkout -b feature/amazing-feature`)
3. 💾 Commit your changes (`git commit -m 'Add some amazing feature'`)
4. 📤 Push to the branch (`git push origin feature/amazing-feature`)
5. 🔄 Open a Pull Request

Please make sure to update tests as appropriate.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors who have helped shape Copilot Arena
- Special thanks to the model providers that make this possible

---
<div align="center">
Made with ❤️ by the Copilot Arena Team  
<br>

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/waynechi?style=flat-square&logo=x&label=Wayne%20Chi)](https://twitter.com/iamwaynechi)
[![GitHub](https://img.shields.io/badge/waynchi-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/waynchi)
[![Website](https://img.shields.io/badge/waynechi.com-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://www.waynechi.com/)

[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/valeriechen_?style=flat-square&logo=x&label=Valerie%20Chen)](https://twitter.com/valeriechen_)
[![GitHub](https://img.shields.io/badge/valeriechen-181717?style=flat-square&logo=github&logoColor=white)](https://github.com/valeriechen)
[![Website](https://img.shields.io/badge/valeriechen.github.io-4285F4?style=flat-square&logo=google-chrome&logoColor=white)](https://valeriechen.github.io/)
</div>
