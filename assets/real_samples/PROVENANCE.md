# Real cheque samples: provenance and licensing

Sample cheque/check images for qualitative real-world testing of the MICR OCR
system. Every file here is sourced from Wikimedia Commons under a permissive
license (CC0, CC-BY, or CC-BY-SA). No "all rights reserved" or stock-site
material is included.

All files were downloaded with a descriptive User-Agent
(`Mozilla/5.0 (research; micr-ocr)`), verified with `file` to be a real
JPEG/PNG, and confirmed to be larger than 10 KB. The MICR-line visibility
assessment below is from direct visual inspection of each downloaded image.

MICR note: the "MICR line" of interest is the E-13B magnetic-ink character row
along the bottom of the cheque (routing/transit, account, cheque number,
amount, plus the transit/on-us/amount/dash control symbols). E-13B is a modern
(post-1960s) North American standard, so older historical cheques generally do
not carry it.

## Files

### check_with_micr.jpg
- Source page: https://commons.wikimedia.org/wiki/File:Check_with_MICR.jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/7/76/Check_with_MICR.jpg
- Author: SRI International
- License: CC-BY-SA-3.0 (also dual-licensed GFDL 1.2+); SPDX: `CC-BY-SA-3.0`
- License URL: https://creativecommons.org/licenses/by-sa/3.0
- Dimensions: 2712 x 1635, JPEG, ~1.1 MB
- MICR line visible: YES. Specimen "Mary/Walter Adams" demonstration check
  with a clear E-13B MICR line, plus a separate annotated breakout labeling
  A.B.A. number, branch number, check digit, account number, transaction code,
  and amount. Best single reference image in this set.

### blank_check.jpg
- Source page: https://commons.wikimedia.org/wiki/File:Blank_check.jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/4/43/Blank_check.jpg
- Author: Mario Lurig (Wikimedia user Ucffool)
- License: CC0 1.0 (public domain dedication); SPDX: `CC0-1.0`
- License URL: http://creativecommons.org/publicdomain/zero/1.0/
- Dimensions: 1800 x 900, JPEG, ~91 KB
- MICR line visible: YES. Clean modern blank US check template (fictitious
  "Ms Jane Doe", routing 123456789, account 0987654321, check 1001) with a
  textbook E-13B line including transit and on-us symbols. No real account data.

### blank_check_spanish.jpg
- Source page: https://commons.wikimedia.org/wiki/File:Blank_check_Spanish.jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/6/6b/Blank_check_Spanish.jpg
- Author: Wikimedia user BeSmall322
- License: CC0 1.0 (public domain dedication); SPDX: `CC0-1.0`
- License URL: http://creativecommons.org/publicdomain/zero/1.0/
- Dimensions: 1800 x 900, JPEG, ~183 KB
- MICR line visible: YES. Spanish-language fictitious blank cheque
  ("Primer Banco de Wiki", Barcelona) with a clear E-13B MICR line including
  transit/on-us/dash control symbols. No real account data.

### canadian_cheque_diagram.png
- Source page: https://commons.wikimedia.org/wiki/File:CanadianCheque.svg
- Direct URL (source SVG): https://upload.wikimedia.org/wikipedia/commons/f/fe/CanadianCheque.svg
- Rasterized via: https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/CanadianCheque.svg/1280px-CanadianCheque.svg.png
- Author: Wikimedia users Twotoque and Airodyssey
- License: CC-BY-SA-4.0; SPDX: `CC-BY-SA-4.0`
- License URL: https://creativecommons.org/licenses/by-sa/4.0
- Dimensions: 1280 x 549, PNG (RGBA)
- MICR line visible: YES. Fictitious Canadian cheque diagram ("MR. JOHN JONES",
  "First Bank of Wiki", payable to Wikimedia Foundation) with a very clean,
  high-contrast E-13B MICR line. The source is an SVG diagram; this file is the
  Wikimedia thumbnail renderer's PNG rasterization at 1280 px width.

### ford_gerald_signed_check.jpg
- Source page: https://commons.wikimedia.org/wiki/File:FORD,_Gerald_(signed_check).jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/4/4c/FORD%2C_Gerald_%28signed_check%29.jpg
- Author: National Museum of American History (digitized by Wikimedia user Godot13)
- License: CC-BY-SA-3.0; SPDX: `CC-BY-SA-3.0`
- License URL: https://creativecommons.org/licenses/by-sa/3.0
- Dimensions: 4000 x 1855, JPEG, ~6.4 MB
- MICR line visible: YES. Real 1975 personal check signed by Gerald R. Ford
  (First National Bank of Washington), high resolution, with a clear E-13B MICR
  line (transit number, account, and amount field 0000002500). This is a
  museum-collection historical check, not a live account, so there is no
  active-account exposure.

### check1005.jpg
- Source page: https://commons.wikimedia.org/wiki/File:Check1005.jpg
- Direct URL: https://upload.wikimedia.org/wikipedia/commons/b/b3/Check1005.jpg
- Author: Wikimedia user Czalex (originally uploaded to be-tarask.wikipedia)
- License: CC-BY-SA-2.5; SPDX: `CC-BY-SA-2.5`
- License URL: https://creativecommons.org/licenses/by-sa/2.5
- Dimensions: 442 x 205, JPEG, ~51 KB
- MICR line visible: YES, but small and low-resolution. Real filled-out check
  (Cal Fed, 1999); the payer name is blacked out but the street address and
  amounts remain. The E-13B MICR line is present and legible to the eye but
  blurry. Useful as a deliberately hard, low-quality real-world stress case;
  weakest image quality in the set.

## Excluded candidate

- `Demand_draft.jpg` (HSBC, CC-BY-SA-3.0, a genuine SPECIMEN draft) was
  downloaded and inspected but removed: its MICR band sits below the
  "Please DO NOT Write Below This Line" marker and is blank in the specimen, so
  no E-13B line is actually printed. Not useful for MICR OCR testing.

## License compliance for redistribution

For the CC-BY-SA and CC-BY files (check_with_micr, canadian_cheque_diagram,
ford_gerald_signed_check, check1005), attribution must be preserved when these
images are shown in the public repo or blog post. Reproduce the author, license
name, and a link back to the source page (the fields above are sufficient). The
CC0 files (blank_check, blank_check_spanish) carry no attribution requirement,
though crediting the source is still polite.
