#!/usr/bin/env python3
"""
Deploy Maritime Domain Awareness System to GitHub

This script helps you deploy the cleaned up repository to GitHub,
either updating an existing repository or creating a new one.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description, check=True, capture_output=False):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=check, 
            capture_output=capture_output, 
            text=True,
            cwd=os.getcwd()
        )
        if capture_output:
            return result.stdout.strip()
        print(f"✅ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        if capture_output and e.stdout:
            print(f"Output: {e.stdout}")
        if capture_output and e.stderr:
            print(f"Error: {e.stderr}")
        return False

def check_git_status():
    """Check if we're in a git repository"""
    if not os.path.exists('.git'):
        print("❌ This is not a git repository")
        return False
    
    # Check if there are any remotes
    result = run_command("git remote -v", "Checking git remotes", check=False, capture_output=True)
    if not result:
        print("⚠️ No git remotes configured")
        return False
    
    print(f"📡 Current remotes:\n{result}")
    return True

def clean_git_history():
    """Clean git history for a fresh start"""
    print("🧹 Cleaning git history for fresh deployment...")
    
    # Remove .git directory and reinitialize
    if os.path.exists('.git'):
        import shutil
        shutil.rmtree('.git')
    
    commands = [
        ("git init", "Initializing new git repository"),
        ("git add .", "Adding all files"),
        ('git commit -m "Initial commit: Maritime Domain Awareness System v1.0.0"', "Creating initial commit"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def setup_git_repository():
    """Set up git repository"""
    print("⚙️ Setting up git repository...")
    
    # Check if already a git repo
    if os.path.exists('.git'):
        print("📁 Git repository already exists")
        
        # Check for uncommitted changes
        result = run_command("git status --porcelain", "Checking for changes", capture_output=True)
        if result:
            print("📝 Found uncommitted changes:")
            run_command("git status", "Showing git status")
            
            response = input("\n🤔 Do you want to commit all changes? (y/N): ").lower()
            if response == 'y':
                run_command("git add .", "Adding all files")
                commit_msg = input("💬 Enter commit message (or press Enter for default): ").strip()
                if not commit_msg:
                    commit_msg = "Update: Cleaned up project structure and organization"
                run_command(f'git commit -m "{commit_msg}"', "Committing changes")
            else:
                print("⏭️ Skipping commit")
    else:
        # Initialize new repository
        if not clean_git_history():
            return False
    
    return True

def push_to_github():
    """Push to GitHub"""
    print("🚀 Deploying to GitHub...")
    
    # Get current remote URL
    remote_url = run_command("git remote get-url origin", "Getting remote URL", check=False, capture_output=True)
    
    if not remote_url:
        print("⚠️ No 'origin' remote found")
        repo_url = input("🌐 Enter your GitHub repository URL (https://github.com/username/repo.git): ").strip()
        if not repo_url:
            print("❌ No repository URL provided")
            return False
        
        if not run_command(f"git remote add origin {repo_url}", "Adding remote origin"):
            return False
        remote_url = repo_url
    
    print(f"📡 Remote URL: {remote_url}")
    
    # Ask about force push
    print("\n⚠️ IMPORTANT: This will completely replace the remote repository!")
    print("   All existing history and files on GitHub will be lost.")
    response = input("🤔 Are you sure you want to force push? (yes/N): ").lower()
    
    if response != 'yes':
        print("❌ Deployment cancelled")
        return False
    
    # Force push to main branch
    if not run_command("git branch -M main", "Setting main branch"):
        return False
    
    if not run_command("git push -f origin main", "Force pushing to GitHub"):
        return False
    
    print("🎉 Successfully deployed to GitHub!")
    print(f"🔗 Repository URL: {remote_url.replace('.git', '')}")
    return True

def create_data_directories():
    """Create necessary data directories with README files"""
    print("📁 Creating data directories...")
    
    directories = {
        "data": "Place your SSDD dataset here",
        "data/SSDD_coco": "SAR Ship Detection Dataset in COCO format",
        "models": "Place your trained YOLO model (best.pt) here",
        "output": "Generated output files will be saved here",
        "results": "Analysis results and reports will be saved here"
    }
    
    for dir_path, description in directories.items():
        os.makedirs(dir_path, exist_ok=True)
        
        readme_path = os.path.join(dir_path, "README.md")
        if not os.path.exists(readme_path):
            with open(readme_path, "w") as f:
                f.write(f"# {dir_path.replace('/', ' / ').title()}\n\n{description}\n\n")
                
                if "models" in dir_path:
                    f.write("## Required Files\n\n")
                    f.write("- `best.pt`: Trained YOLO model for ship detection\n\n")
                    f.write("## Training\n\n")
                    f.write("See `notebooks/YOLO_Ship_Segmentation_Training.ipynb` for model training details.\n")
                
                elif "data" in dir_path:
                    f.write("## Dataset Structure\n\n")
                    f.write("```\n")
                    f.write("SSDD_coco/\n")
                    f.write("├── 000001.jpg\n")
                    f.write("├── 000001.json\n")
                    f.write("├── 000002.jpg\n")
                    f.write("├── 000002.json\n")
                    f.write("└── ...\n")
                    f.write("```\n")
        
        print(f"✅ Created: {dir_path}")

def verify_project_structure():
    """Verify the project has the correct structure"""
    print("🔍 Verifying project structure...")
    
    required_files = [
        "README.md",
        "LICENSE", 
        "requirements.txt",
        "setup.py",
        "maritime_app.py",
        "src/__init__.py",
        "src/core/maritime_tracking_system.py",
        "src/dashboard/maritime_dashboard.py",
        "examples/simple_maritime_demo.py"
    ]
    
    required_dirs = [
        "src/core",
        "src/dashboard", 
        "src/preprocessing",
        "examples",
        "tests",
        "docs",
        "scripts",
        "notebooks"
    ]
    
    missing_files = []
    missing_dirs = []
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            missing_dirs.append(dir_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
    
    if missing_dirs:
        print("❌ Missing required directories:")
        for dir in missing_dirs:
            print(f"   - {dir}")
    
    if missing_files or missing_dirs:
        return False
    
    print("✅ Project structure verification passed")
    return True

def main():
    """Main deployment function"""
    print("🚢 Maritime Domain Awareness System - GitHub Deployment")
    print("=" * 60)
    
    # Verify project structure
    if not verify_project_structure():
        print("\n❌ Project structure verification failed!")
        print("Please ensure all required files and directories are present.")
        return False
    
    # Create data directories
    create_data_directories()
    
    # Setup git repository
    if not setup_git_repository():
        print("\n❌ Git repository setup failed!")
        return False
    
    # Check git status
    if not check_git_status():
        print("\n❌ Git status check failed!")
        return False
    
    # Push to GitHub
    if not push_to_github():
        print("\n❌ GitHub deployment failed!")
        return False
    
    print("\n" + "=" * 60)
    print("🎉 DEPLOYMENT SUCCESSFUL!")
    print("=" * 60)
    print("\n📋 Next Steps:")
    print("1. 📁 Add your trained YOLO model to models/best.pt")
    print("2. 📊 Add your SSDD dataset to data/SSDD_coco/")
    print("3. 🧪 Test the system: python maritime_app.py setup")
    print("4. 🚀 Launch dashboard: python maritime_app.py dashboard")
    print("\n✨ Your Maritime Domain Awareness System is now live on GitHub!")
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Deployment cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)