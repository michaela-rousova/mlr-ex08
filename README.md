# Leaf Classifier (MLR exercise 08)

Live leaf classification in the browser: rear camera (or a photo upload as fallback), ONNX model running via onnxruntime-web. Single file: `index.html`.

**Open the page here:** https://michaela-rousova.github.io/mlr-ex08/

Point your phone camera at a leaf of one of the four classes (apple, cherry, chestnut, maple) and the probabilities update live. A QR code at the bottom of the page encodes the URL above.

## Files

- `index.html` — the whole app (camera, upload fallback, live probabilities, locally generated QR code)
- `leaf_model.onnx` — the classifier (input `image`: 1x3x96x96 float32; output `logits`: 1x4)
- `model_info.json` — classes, input size, channel mean/std
- `check_image.jpg` + `check_expected.json` — verification pair; append `?selftest=1` to the page URL to run `check_image.jpg` through the exact browser pipeline and compare (title shows `SELFTEST_DONE maxdiff=...`)

Preprocessing follows `model_info.json`: central-square crop, resize to 96x96, RGB scaled to 0-1, per-channel standardization with mean/std, input shape 1x3x96x96, softmax over the output logits.

Works in Safari on iPhone (single-threaded WASM, since GitHub Pages cannot send cross-origin-isolation headers; camera requires https).
