# Contributing to RM-Gallery

## Welcome! 🎉

Thank you for your interest in contributing to RM-Gallery! As an open-source project, we warmly welcome and encourage
contributions from the community. Whether you're fixing bugs, adding new features, improving documentation, or sharing
ideas, your contributions help make RM-Gallery better for everyone.

## How to Contribute

To ensure smooth collaboration and maintain the quality of the project, please follow these guidelines when contributing:

### 1. Check Existing Plans and Issues

Before starting your contribution, please review our development roadmap:

- Check the Issues to see our planned development tasks.

  - **If a related issue exists** and is marked as unassigned or open:
    - Please comment on the issue to express your interest in working on it
    - This helps avoid duplicate efforts and allows us to coordinate development

  - **If no related issue exists**:
    - Please create a new issue describing your proposed changes or feature
    - Our team will respond promptly to provide feedback and guidance
    - This helps us maintain the project roadmap and coordinate community efforts

### 2. Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification. This leads to more readable
commit history and enables automatic changelog generation.

**Format:**
```
<type>(<scope>): <subject>
```

**Types:**
- `feat:` A new feature
- `fix:` A bug fix
- `docs:` Documentation only changes
- `style:` Changes that do not affect the meaning of the code (white-space, formatting, etc)
- `refactor:` A code change that neither fixes a bug nor adds a feature
- `perf:` A code change that improves performance
- `ci:` Adding missing tests or correcting existing tests
- `chore:` Changes to the build process or auxiliary tools and libraries

**Examples:**
```bash
feat(graders): add support for custom rubric grading
fix(runner): resolve concurrency issue in batch processing
docs(readme): update installation instructions
refactor(schema): simplify grader result structure
ci(tests): add unit tests for multimodal graders
```

### 3. Code Development Guidelines

#### a. Pre-commit Checks

Before submitting code, you must run pre-commit hooks to ensure code quality and consistency:

**Installation:**
```bash
pip install pre-commit
pre-commit install
```

**Running pre-commit:**
```bash
# Run on all files
pre-commit run --all-files

# Pre-commit will automatically run on git commit after installation
```

#### b. Import Statement Guidelines

RM-Gallery follows a **lazy import principle** to minimize resource loading:

- **DO**: Import modules only when they are actually used
  ```python
  def some_function():
      import openai
      # Use openai library here
  ```

This approach ensures that `import rm_gallery` remains lightweight and doesn't load unnecessary dependencies.

#### c. Unit Tests

- All new features must include appropriate unit tests
- Ensure existing tests pass before submitting your PR
- Run tests using:
  ```bash
  pytest tests
  ```

#### d. Documentation

- Update relevant documentation for new features
- Include code examples where appropriate
- Update the README.md if your changes affect user-facing functionality

## Types of Contributions

### Adding New Graders

RM-Gallery currently supports various types of graders including text, code, math, multimodal, and agent graders.

To add a new grader:

1. **Determine the category**: Choose the most appropriate category (text, code, math, multimodal, agent, or format) for your grader
2. **Create the grader class**:
   ```python
   from rm_gallery.core.graders.base_grader import BaseGrader

   class YourNewGrader(BaseGrader):
       """
       Implementation of your new grader
       """
   ```
3. **Add appropriate tests**: Create unit tests in the corresponding test directory
4. **Update documentation**: Add documentation for your new grader

### Adding New Generators

Generators in RM-Gallery are responsible for automatically creating graders from data:

1. **Create the generator class**:
   ```python
   from rm_gallery.core.generator.base_generator import BaseGenerator

   class YourNewGenerator(BaseGenerator):
       """
       Implementation of your new generator
       """
   ```
2. **Add appropriate tests**: Create unit tests in the generator test directory
3. **Update documentation**: Document your new generator's functionality

### Adding New Analyzers

Analyzers in RM-Gallery are used to analyze grader results:

1. **Create the analyzer class**:
   ```python
   from rm_gallery.core.analyzer.base_analyzer import BaseAnalyzer

   class YourNewAnalyzer(BaseAnalyzer):
       """
       Implementation of your new analyzer
       """
   ```
2. **Add appropriate tests**: Create unit tests in the analyzer test directory
3. **Update documentation**: Document your new analyzer's functionality

### Adding New Models

RM-Gallery supports various models for grading:

1. **Create the model class**:
   ```python
   from rm_gallery.core.models.base_chat_model import BaseChatModel

   class YourNewModel(BaseChatModel):
       """
       Implementation of your new model
       """
   ```
2. **Add appropriate tests**: Create unit tests in the model test directory
3. **Update documentation**: Document your new model's functionality

## Do's and Don'ts

### ✅ DO:

- **Start small**: Begin with small, manageable contributions
- **Communicate early**: Discuss major changes before implementing them
- **Write tests**: Ensure your code is well-tested
- **Document your code**: Help others understand your contributions
- **Follow commit conventions**: Use conventional commit messages
- **Be respectful**: Follow our Code of Conduct
- **Ask questions**: If you're unsure about something, just ask!

### ❌ DON'T:

- **Don't surprise us with big pull requests**: Large, unexpected PRs are difficult to review and may not align with project goals. Always open an issue first to discuss major changes
- **Don't ignore CI failures**: Fix any issues flagged by continuous integration
- **Don't mix concerns**: Keep PRs focused on a single feature or fix
- **Don't forget to update tests**: Changes in functionality should be reflected in tests
- **Don't break existing APIs**: Maintain backward compatibility when possible, or clearly document breaking changes
- **Don't add unnecessary dependencies**: Keep the core library lightweight
- **Don't bypass the lazy import principle**: This keeps RM-Gallery fast to import

## Getting Help

If you need assistance or have questions:

- 🐛 Report bugs via Issues
- 📧 Contact the maintainers



Thank you for contributing to RM-Gallery! Your efforts help build a better tool for the entire community. 🚀