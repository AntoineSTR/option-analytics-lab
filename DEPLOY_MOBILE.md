# Mobile version (Android & iPhone)

The app is a Streamlit web app: once deployed it runs in the phone's browser, adapts to the screen width, and can be pinned to the home screen with its own icon, like a native app. Nothing to install from a store, and the same URL works on Android, iPhone, tablets and computers.

## Step 1 - Put the code on GitHub

1. Create a free account on https://github.com (if needed).
2. Create a new **public** repository, e.g. `option-analytics-lab`.
3. Upload every file of this folder (drag and drop on the repository page, "Add file → Upload files"), keeping the folders `core/`, `views/`, `ui/`, `data/`, `.streamlit/`.
   `app.py` and `requirements.txt` must be at the root of the repository.

## Step 2 - Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and sign in with GitHub.
2. Click **Create app → Deploy a public app from GitHub**.
3. Repository: `your-name/option-analytics-lab`, Branch: `main`, Main file path: `app.py`.
4. Optional: choose a custom subdomain, e.g. `option-analytics-lab` → `https://option-analytics-lab.streamlit.app`.
5. Click **Deploy**. The first build takes 2-4 minutes (it installs `requirements.txt`).

Every new commit on GitHub redeploys the app automatically.

## Step 3 - Install it on the phone

**iPhone (Safari)**: open the URL → Share button → **Add to Home Screen** → Add.
**Android (Chrome)**: open the URL → ⋮ menu → **Add to Home screen** (or **Install app**) → Add.

The app then opens full screen from its icon.

## Notes

- Free apps go to sleep after a few days without visits: the first visit shows "wake up this app" (≈30 s).
- Live prices come from Yahoo Finance; if the cloud server cannot reach it at some point, enter the spot manually - every module keeps working.
- On a phone: the ☰ / › button at the top-left opens the module menu; wide tables scroll horizontally; charts can be zoomed and saved (camera icon).
