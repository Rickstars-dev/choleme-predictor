# Quick Deploy Script for CholeME
# Run this in PowerShell to prepare for deployment

Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "CholeME - Deployment Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""

# Check if git is installed
Write-Host "Checking for Git..." -ForegroundColor Yellow
$gitInstalled = Get-Command git -ErrorAction SilentlyContinue

if (-not $gitInstalled) {
    Write-Host "❌ Git is not installed!" -ForegroundColor Red
    Write-Host "Please install Git from: https://git-scm.com/download/win" -ForegroundColor Yellow
    Write-Host "Then run this script again." -ForegroundColor Yellow
    exit
}

Write-Host "✅ Git is installed" -ForegroundColor Green
Write-Host ""

# Get GitHub username
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Step 1: GitHub Setup" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
$username = Read-Host "Enter your GitHub username"

Write-Host ""
Write-Host "Great! Your repository will be at:" -ForegroundColor Green
Write-Host "https://github.com/$username/choleme-predictor" -ForegroundColor Cyan
Write-Host ""

# Initialize git
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Step 2: Initializing Git" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan

if (Test-Path ".git") {
    Write-Host "Git already initialized" -ForegroundColor Yellow
} else {
    git init
    Write-Host "✅ Git initialized" -ForegroundColor Green
}

# Create .gitignore if not exists
if (-not (Test-Path ".gitignore")) {
    @"
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
.env
.venv
*.log
instance/
.webassets-cache
.pytest_cache/
.coverage
htmlcov/
dist/
build/
*.egg-info/
.DS_Store
"@ | Out-File -FilePath ".gitignore" -Encoding UTF8
    Write-Host "✅ Created .gitignore" -ForegroundColor Green
}

# Add all files
Write-Host ""
Write-Host "Adding files to Git..." -ForegroundColor Yellow
git add .
Write-Host "✅ Files added" -ForegroundColor Green

# Commit
Write-Host ""
Write-Host "Creating commit..." -ForegroundColor Yellow
git commit -m "Initial commit - CholeME cholesterol predictor by Abhishek Chandra"
Write-Host "✅ Commit created" -ForegroundColor Green

# Set main branch
git branch -M main

# Add remote
Write-Host ""
Write-Host "Adding GitHub remote..." -ForegroundColor Yellow
$remoteUrl = "https://github.com/$username/choleme-predictor.git"
git remote remove origin 2>$null
git remote add origin $remoteUrl
Write-Host "✅ Remote added: $remoteUrl" -ForegroundColor Green

Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Next Steps:" -ForegroundColor Cyan
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor Yellow
Write-Host "   https://github.com/new" -ForegroundColor Cyan
Write-Host "   Name: choleme-predictor" -ForegroundColor White
Write-Host "   Make it PUBLIC (for free deployment)" -ForegroundColor White
Write-Host ""
Write-Host "2. Then run this command to push:" -ForegroundColor Yellow
Write-Host "   git push -u origin main" -ForegroundColor Cyan
Write-Host ""
Write-Host "3. Deploy on Render:" -ForegroundColor Yellow
Write-Host "   https://render.com" -ForegroundColor Cyan
Write-Host "   - Sign in with GitHub" -ForegroundColor White
Write-Host "   - New + → Web Service" -ForegroundColor White
Write-Host "   - Select choleme-predictor repo" -ForegroundColor White
Write-Host "   - Start Command: gunicorn app:app" -ForegroundColor White
Write-Host "   - Click Create Web Service" -ForegroundColor White
Write-Host ""
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host "Your app will be live in 2-3 minutes!" -ForegroundColor Green
Write-Host "=====================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "For detailed instructions, see DEPLOYMENT_GUIDE.md" -ForegroundColor Yellow
