#!/usr/bin/env python3
"""
GitHub Action script to parse docs/development-roadmap.md and create GitHub issues
for actionable tasks marked with checkboxes.
"""

import os
import re
import sys
import json
import requests
from typing import List, Dict, Tuple, Optional


class RoadmapParser:
    def __init__(self, github_token: str, repository: str, dry_run: bool = False):
        self.github_token = github_token
        self.repository = repository
        self.dry_run = dry_run
        self.api_base = f"https://api.github.com/repos/{repository}"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
    def read_roadmap(self) -> str:
        """Read the development roadmap file."""
        roadmap_path = "docs/development-roadmap.md"
        try:
            with open(roadmap_path, 'r', encoding='utf-8') as f:
                return f.read()
        except FileNotFoundError:
            print(f"Error: Could not find {roadmap_path}")
            sys.exit(1)
    
    def parse_roadmap_structure(self, content: str, phase_filter: str = "") -> List[Dict]:
        """Parse the roadmap markdown and extract structured information."""
        lines = content.split('\n')
        issues = []
        current_phase = ""
        current_section = ""
        current_subsection = ""
        current_objectives = []
        current_context = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip if we have a phase filter and this isn't the right phase
            if phase_filter and current_phase and phase_filter.lower() not in current_phase.lower():
                continue
            
            # Phase headers (## Phase X:)
            if re.match(r'^## Phase \d+:', line):
                current_phase = line.replace('##', '').strip()
                current_section = ""
                current_subsection = ""
                current_objectives = []
                current_context = []
                continue
            
            # Section headers (### X.X)
            if re.match(r'^### \d+\.\d+', line):
                current_section = line.replace('###', '').strip()
                current_subsection = ""
                current_objectives = []
                current_context = []
                continue
            
            # Subsection headers (#### or #####)
            if re.match(r'^####+ ', line):
                current_subsection = line.replace('#', '').strip()
                current_objectives = []
                current_context = []
                continue
            
            # Objectives section
            if line == "#### Objectives":
                # Collect objectives until we hit actionable steps
                obj_lines = []
                j = i + 1
                while j < len(lines) and not lines[j].strip().startswith("#### Actionable Steps"):
                    if lines[j].strip() and not lines[j].strip().startswith('#'):
                        obj_lines.append(lines[j].strip())
                    j += 1
                current_objectives = obj_lines
                continue
            
            # Actionable tasks (checkboxes)
            if re.match(r'^- \[ \]', line):
                task_text = line.replace('- [ ]', '').strip()
                
                if not task_text:
                    continue
                
                # Collect context from surrounding lines
                context_lines = []
                
                # Look back for relevant context
                for k in range(max(0, i-5), i):
                    context_line = lines[k].strip()
                    if context_line and not context_line.startswith('#') and not context_line.startswith('-'):
                        context_lines.append(context_line)
                
                # Look forward for additional context (like code blocks)
                for k in range(i+1, min(len(lines), i+10)):
                    context_line = lines[k].strip()
                    if context_line.startswith('```') or context_line.startswith('- [ ]'):
                        break
                    if context_line and not context_line.startswith('#'):
                        context_lines.append(context_line)
                
                issue_data = {
                    'task': task_text,
                    'phase': current_phase,
                    'section': current_section,
                    'subsection': current_subsection,
                    'objectives': current_objectives,
                    'context': context_lines,
                    'line_number': i + 1
                }
                
                issues.append(issue_data)
        
        return issues
    
    def group_issues(self, issues: List[Dict]) -> List[Dict]:
        """Group related issues into meaningful GitHub issues."""
        grouped = {}
        
        for issue in issues:
            # Create a grouping key based on section and subsection
            section_key = f"{issue['phase']} - {issue['section']}"
            if issue['subsection']:
                section_key += f" - {issue['subsection']}"
            
            if section_key not in grouped:
                grouped[section_key] = {
                    'title': section_key,
                    'phase': issue['phase'],
                    'section': issue['section'],
                    'subsection': issue['subsection'],
                    'objectives': issue['objectives'],
                    'tasks': [],
                    'context': set()
                }
            
            grouped[section_key]['tasks'].append(issue['task'])
            grouped[section_key]['context'].update(issue['context'])
        
        # Convert to list and clean up context
        result = []
        for group_key, group_data in grouped.items():
            group_data['context'] = list(group_data['context'])
            result.append(group_data)
        
        return result
    
    def create_issue_body(self, group: Dict) -> str:
        """Create the GitHub issue body from grouped data."""
        body_parts = []
        
        # Add phase and section information
        body_parts.append(f"**Phase:** {group['phase']}")
        body_parts.append(f"**Section:** {group['section']}")
        if group['subsection']:
            body_parts.append(f"**Subsection:** {group['subsection']}")
        
        body_parts.append("")
        
        # Add objectives
        if group['objectives']:
            body_parts.append("## Objectives")
            for obj in group['objectives']:
                if obj.startswith('-'):
                    body_parts.append(obj)
                else:
                    body_parts.append(f"- {obj}")
            body_parts.append("")
        
        # Add actionable tasks
        body_parts.append("## Actionable Tasks")
        for task in group['tasks']:
            body_parts.append(f"- [ ] {task}")
        
        body_parts.append("")
        
        # Add context if available
        if group['context']:
            body_parts.append("## Additional Context")
            for context in group['context']:
                if context and not context.startswith('```'):
                    body_parts.append(context)
                elif context.startswith('```'):
                    body_parts.append(context)
            body_parts.append("")
        
        # Add reference to roadmap
        body_parts.append("---")
        body_parts.append("*This issue was automatically generated from [docs/development-roadmap.md](../blob/main/docs/development-roadmap.md)*")
        
        return "\n".join(body_parts)
    
    def get_phase_label(self, phase: str) -> str:
        """Extract phase number for labeling."""
        match = re.search(r'Phase (\d+)', phase)
        if match:
            return f"phase-{match.group(1)}"
        return "roadmap"
    
    def create_github_issue(self, group: Dict) -> bool:
        """Create a GitHub issue for the given group."""
        title = group['title']
        body = self.create_issue_body(group)
        
        # Create labels
        labels = ["roadmap", "enhancement"]
        phase_label = self.get_phase_label(group['phase'])
        if phase_label != "roadmap":
            labels.append(phase_label)
        
        issue_data = {
            "title": title,
            "body": body,
            "labels": labels
        }
        
        if self.dry_run:
            print(f"\n{'='*60}")
            print(f"DRY RUN - Would create issue:")
            print(f"Title: {title}")
            print(f"Labels: {', '.join(labels)}")
            print(f"Body preview (first 200 chars):")
            print(body[:200] + "..." if len(body) > 200 else body)
            print(f"Total tasks: {len(group['tasks'])}")
            return True
        
        # Create the issue
        response = requests.post(
            f"{self.api_base}/issues",
            headers=self.headers,
            json=issue_data
        )
        
        if response.status_code == 201:
            issue_url = response.json()['html_url']
            print(f"✅ Created issue: {title}")
            print(f"   URL: {issue_url}")
            print(f"   Tasks: {len(group['tasks'])}")
            return True
        else:
            print(f"❌ Failed to create issue: {title}")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
    
    def run(self, phase_filter: str = "") -> None:
        """Main execution function."""
        print("🚀 Starting roadmap-to-issues transformation")
        print(f"Repository: {self.repository}")
        print(f"Dry run: {self.dry_run}")
        if phase_filter:
            print(f"Phase filter: {phase_filter}")
        print()
        
        # Read and parse roadmap
        print("📖 Reading development roadmap...")
        content = self.read_roadmap()
        
        print("🔍 Parsing roadmap structure...")
        issues = self.parse_roadmap_structure(content, phase_filter)
        print(f"Found {len(issues)} actionable tasks")
        
        if not issues:
            print("No actionable tasks found. Exiting.")
            return
        
        print("📦 Grouping related tasks...")
        grouped_issues = self.group_issues(issues)
        print(f"Grouped into {len(grouped_issues)} issues")
        
        # Create issues
        print("\n🏗️  Creating GitHub issues...")
        success_count = 0
        for group in grouped_issues:
            if self.create_github_issue(group):
                success_count += 1
        
        print(f"\n✨ Summary:")
        if self.dry_run:
            print(f"   Would create {len(grouped_issues)} issues")
            print(f"   Total actionable tasks: {len(issues)}")
        else:
            print(f"   Successfully created: {success_count}/{len(grouped_issues)} issues")
            print(f"   Total actionable tasks: {len(issues)}")


def main():
    """Main entry point."""
    # Get environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    repository = os.getenv('GITHUB_REPOSITORY')
    dry_run = os.getenv('DRY_RUN', 'false').lower() == 'true'
    phase_filter = os.getenv('PHASE_FILTER', '').strip()
    
    if not github_token:
        print("Error: GITHUB_TOKEN environment variable is required")
        sys.exit(1)
    
    if not repository:
        print("Error: GITHUB_REPOSITORY environment variable is required")
        sys.exit(1)
    
    # Create parser and run
    parser = RoadmapParser(github_token, repository, dry_run)
    try:
        parser.run(phase_filter)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()