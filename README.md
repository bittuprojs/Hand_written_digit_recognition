
# Handwritten Digit Recognition – Streamlit (Live Drawing Pad)

## Quick start
1) Install dependencies
```bash
pip install -r requirements.txt
```

2) Train the model (creates `mnist_cnn_model.h5`)
```bash
python train_model.py
```

3) Run the app
```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud
- Push these files to a GitHub repo.
- On https://share.streamlit.io, create an app pointing to `app.py`.
- Add these secrets (if needed): none required.
- Hardware note: TensorFlow is a heavy dependency; first build may take a while.

## Notes
- The canvas draws black on white. The app inverts, centers, and pads to match MNIST (white digit on black background).
- For best accuracy: write centrally with thick strokes.
