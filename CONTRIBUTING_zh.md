# 贡献到 RM-Gallery

## 欢迎！🎉

感谢开源社区对 RM-Gallery 项目的关注和支持，作为一个开源项目，我们热烈欢迎并鼓励来自社区的贡献。无论是修复错误、添加新功能、改进文档还是
分享想法，这些贡献都能帮助 RM-Gallery 变得更好。

## 如何贡献

为了确保顺利协作并保持项目质量，请在贡献时遵循以下指南：

### 1. 检查现有计划和问题

在开始贡献之前，请查看我们的开发任务：

- 查看 Issues 以了解我们计划的开发任务。

  - **如果存在相关问题** 并且标记为未分配或开放状态：
    - 请在该问题下评论，表达您有兴趣参与该任务
    - 这有助于协调开发工作，避免重复工作

  - **如果不存在相关问题**：
    - 请创建一个新 issue 用以描述对应的更改或功能
    - 我们的团队将及时进行回复并提供反馈
    - 这有助于我们维护项目路线图并协调社区工作

### 2. 提交信息格式

RM-Gallery 遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范。这使得提交历史更易读，并能够自动生成更新日志。

**格式：**
```
<type>(<scope>): <subject>
```

**类型：**
- `feat:` 新功能
- `fix:` 错误修复
- `docs:` 仅文档更改
- `style:` 不影响代码含义的更改（空格、格式等）
- `refactor:` 既不修复错误也不添加功能的代码更改
- `perf:` 提高性能的代码更改
- `ci:` 添加缺失的测试或更正现有测试
- `chore:` 对构建过程或辅助工具和库的更改

**示例：**
```bash
feat(graders): add support for custom rubric grading
fix(runner): resolve concurrency issue in batch processing
docs(readme): update installation instructions
refactor(schema): simplify grader result structure
ci(tests): add unit tests for multimodal graders
```

### 3. 代码开发指南

#### a. 提交前检查

在提交代码之前，请运行 pre-commit 钩子以确保代码质量和一致性：

**安装：**
```bash
pip install pre-commit
pre-commit install
```

**运行 pre-commit：**
```bash
# 在所有文件上运行
pre-commit run --all-files

# 安装后，pre-commit 将在 git commit 时自动运行
```

#### b. 关于代码中的 Import

RM-Gallery 遵循**懒加载导入原则**以最小化资源加载：

- **推荐做法**：仅在实际使用时导入模块
  ```python
  def some_function():
      import openai
      # 在此处使用 openai 库
  ```

这种方法确保 `import rm_gallery` 是一个轻量操作，不会加载不必要的依赖项。

#### c. 单元测试

- 所有新功能都必须包含适当的单元测试
- 在提交 PR 之前确保现有测试通过
- 使用以下命令运行测试：
  ```bash
  pytest tests
  ```

#### d. 文档

- 为新功能更新相关文档
- 在适当的地方包含代码示例
- 如果更改影响面向用户的功能，请更新 README.md

## 贡献类型

### 添加新的评估器 (Graders)

RM-Gallery 目前支持多种类型的评估器，包括文本、代码、数学、多模态和智能体评估器。

要添加一个新的评估器：

1. **确定类别**：为你的评估器选择最合适的类别（文本、代码、数学、多模态、智能体或格式）
2. **创建评估器类**：
   ```python
   from rm_gallery.core.graders.base_grader import BaseGrader

   class YourNewGrader(BaseGrader):
       """
       你的新评估器的实现
       """
   ```
3. **添加适当的测试**：在相应的测试目录中创建单元测试
4. **更新文档**：为你的新评估器添加文档

### 添加新的生成器 (Generators)

RM-Gallery 中的生成器负责从数据中自动创建评估器：

1. **创建生成器类**：
   ```python
   from rm_gallery.core.generator.base_generator import BaseGenerator

   class YourNewGenerator(BaseGenerator):
       """
       你的新生成器的实现
       """
   ```
2. **添加适当的测试**：在生成器测试目录中创建单元测试
3. **更新文档**：为你的新生成器添加文档

### 添加新的分析器 (Analyzers)

RM-Gallery 中的分析器用于分析评估器结果：

1. **创建分析器类**：
   ```python
   from rm_gallery.core.analyzer.base_analyzer import BaseAnalyzer

   class YourNewAnalyzer(BaseAnalyzer):
       """
       你的新分析器的实现
       """
   ```
2. **添加适当的测试**：在分析器测试目录中创建单元测试
3. **更新文档**：为你的新分析器添加文档

### 添加新的模型 (Models)

RM-Gallery 支持各种用于评估的模型：

1. **创建模型类**：
   ```python
   from rm_gallery.core.models.base_chat_model import BaseChatModel

   class YourNewModel(BaseChatModel):
       """
       你的新模型的实现
       """
   ```
2. **添加适当的测试**：在模型测试目录中创建单元测试
3. **更新文档**：为你的新模型添加文档

## Do's and Don'ts

### ✅ DO

- **从小处着手**：从小的、可管理的贡献开始
- **及早沟通**：在实现主要功能之前进行讨论
- **编写测试**：确保代码经过充分测试
- **添加代码注释**：帮助他人理解贡献内容
- **遵循提交约定**：使用约定式提交消息
- **保持尊重**：遵守我们的行为准则
- **提出问题**：如果不确定某事，请提问！

### ❌ DON'T

- **不要用大型 PR 让我们措手不及**：大型的、意外的 PR 难以审查，并且可能与项目目标不一致。在进行重大更改之前，请务必先开启一个问题进行讨论
- **不要忽略 CI 失败**：修复持续集成标记的任何问题
- **不要混合关注点**：保持 PR 专注于单一功能的实现或修复
- **不要忘记更新测试**：功能的更改应反映在测试中
- **不要破坏现有 API**：在可能的情况下保持向后兼容性，或清楚地记录破坏性更改
- **不要添加不必要的依赖项**：保持核心库轻量级
- **不要绕过懒加载导入原则**：确保 RM-Gallery 在导入阶段不至于臃肿

## 获取帮助

如果需要帮助或有疑问：

- 🐛 通过 Issues 报告错误
- 📧 联系维护人员



感谢为 RM-Gallery 做出贡献！🚀