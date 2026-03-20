# Deployment Guide - AI Proctor System

This guide covers how to share your project and deploy it for demos or production.

## 1. Pushing to GitHub (or other Git accounts)

Your project is now initialized as a Git repository. To push it to your account:

1. Create a **New Repository** on GitHub/GitLab. Do **not** initialize it with a README or .gitignore (since we already have them).
2. Copy the **Remote URL** (e.g., `https://github.com/yourusername/your-repo.git`).
3. Run these commands in your terminal:
   ```bash
   git remote add origin YOUR_REMOTE_URL
   git branch -M main
   git push -u origin main
   ```

> [!NOTE]
> The `.gitignore` file already excludes your virtual environment and local user data to keep your repo clean.

---

## 2. Shared Deployment Options

### Option A: Local Network (Same WiFi)
Good for testing in a lab or classroom.
1. Find your **Local IP Address**:
   - Run `ipconfig` in CMD. Look for `IPv4 Address` (e.g., `192.168.1.15`).
2. Run the app: `python app.py`.
3. Others can join by typing `http://192.168.1.15:5000` in their browsers.

### Option B: Ngrok (Internet Access)
Best for remote demos or sharing with a client.
1. Download [ngrok](https://ngrok.com/).
2. Run `ngrok http 5000`.
3. Copy the `Forwarding` URL (e.g., `https://abcd-123.ngrok.io`).
4. Anyone in the world can now access your app via that link.

---

## 3. Cloud Hosting (Advanced)

If you plan to deploy to **Heroku, Render, or Vercel**:

> [!WARNING]
> **Hardware Compatibility Issue**:
> Regular cloud servers do NOT have cameras or microphones.
> - The **Vision** and **Audio** modules will show as "Offline".
> - The **OS Monitor** (Keyboard Hook) will NOT work on a remote web server.

**To fix this for production:**
You must rewrite the sensing logic in **JavaScript** (Browser-side) instead of Python (Server-side) so that the student's own browser handles the camera and audio.
- Use `getUserMedia()` for camera/audio.
- Use MediaPipe JS for AI detection.
