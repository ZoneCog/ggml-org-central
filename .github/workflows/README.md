# Roadmap to Issues GitHub Action

This GitHub Action automatically transforms the development roadmap (`docs/development-roadmap.md`) into actionable GitHub issues. It parses the markdown file, extracts tasks marked with checkboxes (`- [ ]`), and creates organized GitHub issues with proper labeling and context.

## How it works

The action:

1. **Parses** the development roadmap markdown file
2. **Extracts** actionable tasks marked with `- [ ]` checkboxes
3. **Groups** related tasks by phase, section, and subsection
4. **Creates** GitHub issues with:
   - Descriptive titles based on the roadmap structure
   - Comprehensive issue bodies including objectives and context
   - Proper labels (`roadmap`, `enhancement`, `phase-X`)
   - Task lists that can be checked off as work progresses

## Usage

### Manual Trigger

The action can be triggered manually from the GitHub Actions tab:

1. Go to **Actions** → **Transform Roadmap to Issues**
2. Click **Run workflow**
3. Optional parameters:
   - **Dry run**: Preview what would be created without actually creating issues
   - **Phase filter**: Only process specific phases (e.g., "Phase 1", "Phase 2")

### Example Workflow Runs

```bash
# Preview all issues that would be created
DRY_RUN: true

# Create issues only for Phase 1
PHASE_FILTER: "Phase 1"

# Create all issues for real
DRY_RUN: false
PHASE_FILTER: ""
```

## Generated Issue Structure

Each generated issue includes:

### Title Format
`Phase X: [Phase Name] - X.Y [Section Name] - [Subsection Name]`

Example: `Phase 1: Advanced Reasoning Engine (Target: Q2 2025) - 1.1 Probabilistic Logic Networks (PLN) Integration - Week 1-2: PLN Foundation`

### Issue Body
- **Phase information** and timeline
- **Section and subsection** context
- **Objectives** from the roadmap
- **Actionable tasks** as checkboxes
- **Additional context** when available
- **Reference link** back to the roadmap

### Labels
- `roadmap` - All generated issues
- `enhancement` - All are considered enhancements
- `phase-X` - Specific phase number (e.g., `phase-1`, `phase-2`)

## Statistics

From the current roadmap:
- **Total actionable tasks**: 148
- **Generated issues**: 37 (grouped logically)
- **Phases covered**: 5 major development phases
- **Average tasks per issue**: 4

## Benefits

1. **Trackable Progress**: Convert roadmap checkboxes into trackable GitHub issues
2. **Project Management**: Use GitHub's project boards, milestones, and assignees
3. **Team Collaboration**: Comment, discuss, and collaborate on specific tasks
4. **Progress Visibility**: See completion status across the entire roadmap
5. **Automated Organization**: Consistent labeling and grouping
6. **Selective Processing**: Create issues for specific phases as needed

## Files

- `.github/workflows/roadmap-to-issues.yml` - GitHub Action workflow
- `.github/scripts/roadmap-parser.py` - Python script that does the parsing and issue creation
- `docs/development-roadmap.md` - Source roadmap file

## Development

The parser script can be run locally for testing:

```bash
# Test parsing (dry run)
cd /path/to/repo
GITHUB_TOKEN=fake_token GITHUB_REPOSITORY=test/repo DRY_RUN=true python3 .github/scripts/roadmap-parser.py

# Test with phase filter
GITHUB_TOKEN=fake_token GITHUB_REPOSITORY=test/repo DRY_RUN=true PHASE_FILTER="Phase 1" python3 .github/scripts/roadmap-parser.py
```

## Safety Features

- **Dry run mode**: Preview what would be created before actual execution
- **Phase filtering**: Process only specific phases to avoid overwhelming issue creation
- **Manual trigger only**: Prevents accidental issue spam
- **Proper error handling**: Graceful failure with informative error messages
- **Duplicate prevention**: Each run creates new issues (manual cleanup if needed)

## Future Enhancements

Potential improvements:
- Issue deduplication (check existing issues before creating)
- Milestone assignment based on phase timelines
- Automatic assignee suggestion based on task type
- Integration with project boards
- Progress synchronization back to roadmap