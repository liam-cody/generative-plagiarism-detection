<div align="center">
<h1>Team JLP at PAN 2026: Span-Based Hybrid Source Retrieval for Generative Plagiarism Detection</h1>

<p>
<a href="https://orcid.org/0009-0002-9221-7655">Liam Kaan Cody</a><sup>1,*</sup>,
Peter Felix Panzitt<sup>1,*</sup>,
James Ernst Cox<sup>1,*</sup>
</p>

<p>
<sup>1</sup> Graz University of Technology<br>
<sup>*</sup> Equal contribution
</p>

<p>
lcody@student.tugraz.at, peter.panzitt@student.tugraz.at, james.cox@student.tugraz.at
</p>
</div>

## 📄 Abstract 

The PAN 2026 generative plagiarism detection task asks systems to recover the source papers used to produce a scientific article written by a large language model. We address this source retrieval problem with a span-based hybrid retrieval system. Instead of representing each suspicious article as one long query, the system decomposes it into sentence-level queries so that local source evidence is not diluted by the full document.

Each sentence is searched against a full-document BM25 index and a dense index built over structure-aware chunks of the source corpus. Sparse and dense rankings are fused with Reciprocal Rank Fusion, and sentence-level evidence is aggregated into a final source-document ranking with CombSUM.

Since no labeled development set was released, we created a 200-query synthetic development set from the source corpus to select span and fusion settings. On this synthetic set, the selected hybrid run reached nDCG@10 of `0.9893` and Recall@10 of `0.9950`. On the official test set of 200 suspicious documents, the primary submission achieved Recall@10 of `0.8542` and nDCG@10 of `0.7994`, ranking first among submitted systems.

## 🔎 Overview 

<div align="center">
  <img alt="System architecture" src="figs/system_architecture.png" width="100%">
  <p><em>Figure 1: System architecture.</em></p>
</div>

The system was built for scientific source retrieval, but most of the pipeline is not specific to scientific papers. The domain-specific part is mainly how the source corpus is indexed and chunked. The same idea, splitting a long query document into local spans, retrieve with sparse and dense methods, then fuse the evidence, should also work for other Query-by-Document retrieval tasks.

## 📚 Citation

[Read the paper](https://ceur-ws.org/Vol-4283/paper400.pdf)

If you use this work, please cite:

```bibtex
@InProceedings{cody:2026,
  author = {Liam Kaan Cody and Peter Felix Panzitt and James Ernst Cox},
  title = {{Team JLP at PAN 2026: Span-Based Hybrid Source Retrieval for Generative Plagiarism Detection}},
  booktitle = {{Working Notes of CLEF 2026 - Conference and Labs of the Evaluation Forum}},
  editor = {Eva S{\'a}nchez Salido and Alberto Barr{\'o}n-Cede{\~n}o and Alba Garc{\'i}a Seco Herrera and Sean MacAvaney and Julia Maria Stru{\ss}},
  month = sep,
  pages = {5439--5449},
  publisher = {CEUR-WS.org},
  venue = {Jena, Germany},
  volume = 4283,
  year = 2026,
  url = {https://ceur-ws.org/Vol-4283/paper400.pdf}
}
```
