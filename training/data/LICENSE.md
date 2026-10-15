# Sentence lists: license and attribution

The files in this folder and `eval/heldout.json` contain text from Wikipedia. They are not covered by the
repository's Apache-2.0 license: they are distributed under the Creative Commons Attribution-ShareAlike 4.0
International license (CC BY-SA 4.0, https://creativecommons.org/licenses/by-sa/4.0/). Share-alike applies to
these lists and to adapted copies of them, not to the code that reads them. They are provided as is, without
warranties (CC BY-SA 4.0, section 5).

| file | content | source and attribution |
|---|---|---|
| `wikitext_sents.json` | 12,000 English sentences | WikiText-103 (Merity, Xiong, Bradbury and Socher, "Pointer Sentinel Mixture Models", arXiv:1609.07843, 2016; CC BY-SA), built from English Wikipedia articles by Wikipedia contributors. The list is Paradee's (https://github.com/sahilmahendrakar/paradee, revision `9c8b4d7`, `training/data/wikitext_sents.json`): the same sentences in the same order. |
| `italian_sents.json` | 200 held-out and 12,000 training sentences from 1,203 articles | Italian Wikipedia, snapshot `20231101.it` of the Hugging Face dataset `wikimedia/wikipedia`. The `source` field of each sentence is the URL of its article; the article's history lists its authors. |
| `../../eval/heldout.json` | the 200 held-out sentences per language, with the token ids the teacher data used | the first 200 accepted sentences of `wikitext_sents.json` and the `heldout` list of `italian_sents.json`; attribution as above |

Changes: single sentences were selected from the articles (see `training/README.md`, "Data");
`eval/heldout.json` adds the phoneme token ids.

Wikipedia's text is licensed under CC BY-SA 4.0 (older revisions also under CC BY-SA 3.0 and the GFDL);
CC BY-SA 3.0 allows adaptations under a later version of the same license.
