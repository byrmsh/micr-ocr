# Reading the MICR line on a check, when the printing fights back

The row of blocky numbers along the bottom of a paper check is the MICR line. It carries the
bank routing number, the account number, and the check number, printed in a font called
E-13B. It is designed to be read magnetically, which is why it survives stamps and signatures
crossing over it. Optically it is a different story: hand it to a general OCR engine and you
get nothing back, because E-13B is not in the engine's alphabet, and the moment a handwritten
amount or a signature loops through the band, even a font-aware reader starts to miss
characters.

I built an end-to-end system that reads the E-13B line under that kind of damage: a synthetic
data generator, a recognizer trained from scratch, a band detector, confidence-based routing
to a human when the model is unsure, and an ONNX service deployed behind a Cloudflare Worker.
This post is the honest version of how it works and how well it actually does.

> One thing up front, stated plainly because it matters: every number here is measured on a
> synthetic benchmark I generated myself. There are no real customer checks in the training or
> the metrics (there is a small qualitative test on a handful of public-domain checks at the
> end). E-13B is a constrained 14-glyph alphabet, so none of this is a claim about
> general-document OCR. What it is: a real, trained, deployed pipeline with numbers you can
> reproduce from the repo.

## The font, without a font

E-13B has fourteen glyphs: the digits 0 to 9 and four control symbols (Transit, Amount, On-Us,
Dash) that bracket and separate the fields. The shapes are defined by ISO 1004 on a grid of
0.013-inch modules, which is where the "13" comes from.

I did not want to ship a proprietary or GPL font into a public repo, and I did not want to
hand-draw glyphs that would look subtly wrong to anyone who knows the font. The clean path was
an Open Font License reconstruction of E-13B drawn from the ISO geometry. I vendored those SVGs
and rasterized them once into a small glyph atlas, so training and serving never touch a font
file or an SVG renderer. The digits come out unmistakably E-13B; the four control symbols sit
at their standard positions.

A MICR line is then composed at a strict fixed pitch (real E-13B is eight characters per inch),
with valid field structure: a nine-digit routing number whose ninth digit is a correct ABA
checksum, an account number that sometimes carries a dash, a check number, and on cleared
checks an amount field. The control symbols, not spaces, are the real delimiters, which is how
the parser later recovers fields even when a character is misread.

## Making the data hard on purpose

A clean MICR line is trivial to read. The interesting question is what happens when the scan is
bad, so the generator degrades each band through three tiers (clean, medium, hard) with effects
that compound: skew and shear, Gaussian and motion blur, sensor noise, JPEG artifacts, ink
bleed and erosion, smudges and stamps, lighting gradients, downscaling, and the one the problem
is really about, handwriting strokes laid across the band as cubic-Bezier ink so a signature or
a handwritten amount overlaps the printed digits.

The honesty trap with self-generated data is obvious: if you train and test on the same
generator you tuned, your accuracy number measures how well the model fits your own pipeline,
not how well it reads MICR. So the generator is frozen before any training, and there is a
separate held-out generator configuration the models never train on, with wider sizes,
different ink darkness, unseen paper textures, and an elastic warp that the training family
never sees. Every headline number below is reported on that held-out generator, so it measures
generalization across generator settings rather than memorization.

## The recognizer: a CRNN trained from scratch

For a fixed 14-glyph alphabet, a 300M-parameter vision-language model would be absurd overkill,
and "I fine-tuned a pretrained transformer" is a weaker, more crowded answer than the honest
one. I trained a CRNN with a CTC loss from scratch, reproducing the architecture from Shi et
al. 2015 ("An End-to-End Trainable Neural Network for Image-based Sequence Recognition"): a
convolutional stack that collapses the image to a feature sequence, a bidirectional LSTM over
that sequence, and a CTC head that outputs per-timestep logits over the 14 classes plus a
blank. CTC decoding and confidence aggregation live outside the network, which is what lets the
exported graph be a plain logits-producer.

The channel counts are kept light because the whole thing trains on a 4GB laptop GPU (an RTX
3050). Batches are bucketed by width so padding, and therefore peak activation memory, stays
small. One detail did most of the work for the end-to-end pipeline: a crop-jitter augmentation
that randomly crops each training band somewhere between the tight ink bounding box and the
full margin, never cutting a glyph. That makes the recognizer robust to however much margin the
detector leaves around the band, which is the main gap between reading a clean crop and reading
the output of a real localizer.

Thirty epochs, about thirty-five minutes, final character accuracy 99.2 percent.

## Results, as a ladder

The fair way to show a recognizer is against the strongest classical baseline you can build,
not against a strawman. So there are three rungs. The floor is a non-ML template matcher: it
renders the 14 reference glyphs, binarizes the band, and classifies each fixed-pitch cell by
normalized cross-correlation with a small local search to absorb pitch jitter. This is the real
approach a pre-neural MICR reader would use. The top rung is the CRNN. And as a reference point
for "what does general OCR do," stock Tesseract, which has no E-13B glyphs at all.

Character error rate by tier, in-distribution test split:

| recognizer          | clean | medium | hard  |
|---------------------|-------|--------|-------|
| Tesseract (stock)   | 0.63  | 0.66   | 0.89  |
| template (classical)| 0.10  | 0.65   | 0.81  |
| CRNN                | 0.00  | 0.00   | 0.024 |

Two honest readings of that table. First, stock OCR cannot read this font, full stop; that row
is a labeled data point, not a baseline anyone is "beating." Second, the classical matcher is
genuinely competent on clean fixed-pitch MICR (10 percent CER, mostly single-glyph confusions)
and then collapses the moment skew, overlap, or low contrast breaks the pitch grid. That
collapse is the whole reason a sequence model earns its place: it does not depend on finding
clean character boundaries.

On the held-out generator (the one the model never trained on), the CRNN holds: 91.9 percent of
full lines exactly correct, 99.2 percent character accuracy, with field-level accuracy of 97.4
percent on the routing number and 96.9 percent on the account number. By tier it is essentially
perfect on clean and medium and drops to 76 percent exact-match on hard, which is where the
handwriting-over-print cases live. Those hard cases are the honest ceiling, and they are exactly
the documents you would want a human to confirm, which is the next piece.

## Knowing when it does not know

A reader that is confidently wrong is worse than one that asks for help. The CRNN's CTC softmax
gives a per-character probability, and the sequence confidence is the mean over the emitted
characters. Raw, that signal is poorly calibrated, which is a known CTC behavior: the softmax is
peaky and blank-dominated, so the numbers run optimistic. I fit a single temperature on the
held-out split to pull confidence back toward the real probability of a correct read, then read
an operating threshold off the coverage-accuracy curve.

The raw signal is bunched near 1.0, so out of the box you cannot threshold it usefully. A
temperature of 2.4 spreads the scores enough that the coverage-accuracy tradeoff becomes
usable. Worth being honest about: the temperature barely moves the expected calibration error
(about 0.056 raw, 0.067 after), because this peaky, blank-dominated CTC confidence is genuinely
hard to calibrate in the strict sense. The useful artifact is not a magically calibrated
probability, it is the tradeoff curve. On the held-out generator, 92.1 percent of lines are
correct with no routing; routing the least-confident 6 percent to a human raises auto-accept
accuracy to 96.9 percent; a balanced operating point routes about 13 percent and auto-accepts
the rest at 97.8 percent; and if you need near-certainty you can auto-accept only the most
confident quarter at 99.8 percent and send everything else to review.

The honest claim here is the method, not the threshold. The temperature and the operating point
are fit on a generator I made up; on a client's real check stream they would have to be
re-calibrated. What transfers is the machinery: temperature scaling, a coverage-accuracy curve,
and a routing rule that sends low-confidence reads to a person.

## Finding the band: classical vs learned

The recognizer reads a cropped band, so something has to find the band on a full check. The
serving path uses a classical OpenCV localizer with no machine learning and no extra license
weight: it strips ruled lines and borders, merges the surviving glyphs into a horizontal band
with a morphological close, rejects single-stroke lines by demanding many components, and tightens
to the dense band rows. It is the right tool because the band is the only wide, short row of many
similar marks near the bottom of a check.

I also trained a learned detector, YOLO11n fine-tuned on synthetic checks, both because it is the
honest answer to "have you fine-tuned YOLO" and because it makes the comparison concrete.

<!-- TODO: fill from eval.detect_eval: classical mean IoU / IoU@0.5 vs YOLO mean IoU / mAP50. -->

The trade is the expected one. The classical localizer is free, license-clean, and good enough on
well-formed checks, and it degrades on the skewed, cluttered ones. The learned detector is more
robust across the board. Note on licensing, since it is the kind of thing that bites later:
Ultralytics YOLO is AGPL-3.0, and that is why this repo is AGPL and why the served container does
not import it. The detector is a training-time and comparison artifact; the runtime uses the
classical localizer.

## Serving it

Training is PyTorch on the GPU; serving is none of that. The recognizer is exported to ONNX
(logits only, opset 17, with an onnx-vs-torch parity check on export) and runs under
onnxruntime on CPU. The container is FastAPI plus onnxruntime plus headless OpenCV, with no
torch and no Ultralytics, which keeps the image small and the cold start short. The training and
serving dependencies are split into separate groups precisely so the heavy half never reaches
the image.

It is deployed the same way as my other small services: a Cloudflare Worker as the HTTP front
door routing to a Durable-Object-backed container. The demo page ships pre-computed results for
its sample images, so the first click returns instantly even while the container is waking from
idle. You POST an image (a full check or a cropped band) to `/read` and get back the recognized
line, the parsed fields, a confidence, a route-to-human flag, and the band's bounding box.

<!-- TODO: live URL once deployed. -->

## What this is not

It is worth being precise about the edges. The accuracy numbers are on synthetic data; the
synthetic-to-real gap is real and unmeasured at scale here. As a qualitative check I ran the
pipeline on a handful of public-domain checks from Wikimedia, and

<!-- TODO: one honest sentence on how the real-check qualitative test went. -->

E-13B is a small, rigid alphabet, so none of this transfers to general document OCR as-is, and I
am not claiming it does. The confidence threshold is calibrated on my generator, not a
production distribution. And the hard tier, the genuinely overlapping handwriting, is where the
model is weakest and where the human-review path is meant to catch it.

What I am comfortable claiming is the engineering: a from-scratch CRNN+CTC recognizer, a
fine-tuned YOLO11n detector, a license-clean synthetic data pipeline, calibrated confidence with
human-in-the-loop routing, and an ONNX service deployed on a 4GB budget, all reproducible from
the repo.
