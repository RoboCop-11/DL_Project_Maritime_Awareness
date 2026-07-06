# Contributing to Maritime Domain Awareness System

Thank you for your interest in contributing to the Maritime Domain Awareness System! This document provides guidelines for contributing to the project.

## 🎯 Getting Started

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/maritime-domain-awareness.git
   cd maritime-domain-awareness
   ```

2. **Set Up Development Environment**
   ```bash
   # Install in development mode
   python dev_setup.py --setup --dirs
   
   # Or manually:
   pip install -e .
   pip install -r requirements-dev.txt
   ```

3. **Create Data Directories**
   ```bash
   python dev_setup.py --dirs
   ```

### Project Structure

```
src/                    # Main source code
├── core/              # Core tracking system
├── dashboard/         # Web dashboard
└── preprocessing/     # Image preprocessing

examples/              # Demo scripts and examples
tests/                 # Test suite
scripts/              # Utility scripts
docs/                 # Documentation
notebooks/            # Jupyter notebooks
```

## 🔧 Development Workflow

### Code Quality

1. **Format Code**
   ```bash
   python dev_setup.py --format
   ```

2. **Run Linting**
   ```bash
   python dev_setup.py --lint
   ```

3. **Run Tests**
   ```bash
   python dev_setup.py --test
   ```

### Making Changes

1. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Your Changes**
   - Follow existing code style and patterns
   - Add tests for new functionality
   - Update documentation as needed
   - Ensure all tests pass

3. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   ```

### Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_dashboard.py

# Run with coverage
pytest tests/ --cov=src/
```

### Writing Tests

- Add tests for new functionality in the `tests/` directory
- Follow existing test patterns and naming conventions
- Aim for good test coverage of new code
- Include both unit tests and integration tests

## 📚 Documentation

### Code Documentation

- Use clear, descriptive docstrings for all functions and classes
- Follow Google or NumPy docstring style
- Include parameter types and return value descriptions
- Add usage examples for complex functions

### User Documentation

- Update README.md for new features
- Add examples to the `examples/` directory
- Update the dashboard guide for UI changes
- Create Jupyter notebooks for complex workflows

## 🐛 Bug Reports

When reporting bugs, please include:

1. **Environment Information**
   - Python version
   - Operating system
   - Package versions (`pip list`)

2. **Steps to Reproduce**
   - Clear, step-by-step instructions
   - Sample code or data if applicable
   - Expected vs. actual behavior

3. **Error Messages**
   - Complete error messages and stack traces
   - Log files if available

## 💡 Feature Requests

For feature requests, please provide:

1. **Use Case**: Describe the problem you're trying to solve
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other solutions you've considered
4. **Impact**: Who would benefit from this feature

## 🚀 Pull Request Process

### Before Submitting

1. **Check Requirements**
   - [ ] Tests pass (`python dev_setup.py --test`)
   - [ ] Code is formatted (`python dev_setup.py --format`)
   - [ ] Linting passes (`python dev_setup.py --lint`)
   - [ ] Documentation is updated
   - [ ] CHANGELOG.md is updated (for significant changes)

2. **Test Your Changes**
   - Test on multiple platforms if possible
   - Verify the dashboard still works
   - Check that examples run correctly

### Submitting the PR

1. **Create Pull Request**
   - Use a clear, descriptive title
   - Provide detailed description of changes
   - Link to any related issues
   - Include screenshots for UI changes

2. **PR Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Breaking change
   - [ ] Documentation update
   
   ## Testing
   - [ ] Tests pass
   - [ ] Manual testing completed
   - [ ] Dashboard functionality verified
   
   ## Checklist
   - [ ] Code follows project style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

## 🏷️ Areas for Contribution

### High Priority
- **Performance Optimization**: Improve tracking and detection speed
- **Enhanced Anomaly Detection**: More sophisticated anomaly types
- **Real-time Video Processing**: Live stream capabilities
- **Additional Preprocessing**: New SAR image enhancement techniques

### Medium Priority
- **Dashboard Improvements**: New visualizations and features
- **API Development**: RESTful API for integration
- **Mobile Support**: Responsive dashboard design
- **Cloud Deployment**: Docker and cloud deployment guides

### Good First Issues
- **Documentation**: Improve code comments and user guides
- **Test Coverage**: Add tests for existing functionality
- **Examples**: Create tutorial notebooks and demos
- **Bug Fixes**: Address reported issues

## 🤝 Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Provide constructive feedback
- Focus on what's best for the community

### Getting Help

- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and ideas
- **Documentation**: Check the docs/ directory for guides

### Recognition

Contributors will be acknowledged in:
- README.md contributors section
- CHANGELOG.md for significant contributions
- Release notes for major features

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to the Maritime Domain Awareness System! Your efforts help make maritime surveillance more effective and accessible. 🚢