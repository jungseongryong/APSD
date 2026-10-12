# APSD

Anonymous source distribution for review.

## Prepare the submission archive

Download the source ZIP, extract this script, and run:

```bash
python3 scripts/make_submission_zip.py /path/to/downloaded-source.zip /path/to/submission.zip
```

Submit `submission.zip`. The script preserves file contents and executable modes,
uses a neutral top-level folder, and removes archive comments, extra metadata,
and original timestamps. It rejects hidden files, cache directories, and symbolic
links for separate review. Existing output files are never overwritten.

Source-hosting downloads can contain a commit identifier in their ZIP comment.
Renaming the downloaded ZIP does not remove that identifier. Repacking does not
remove public copies of the code or make repository history anonymous.

See `data/README.md` for data preparation. Model and dataset identifiers used by
the implementation remain necessary for execution.
