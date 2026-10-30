# species_table.csv

Taxonomic identifiers for the 451 species (classes) used in Peter & Kunzelmann, *Optimizing Spectrogram Front-Ends and Training Strategies for Deep Learning-Based Acoustic Insect Classification*.

One row per species: the 459 species of InsectSet459 v1.0 minus the 8 removed by the per-split minimum (Section 3.1 of the paper). Format: UTF-8 CSV, comma-separated, header row. An empty cell means the value is not available in that source.

## Columns

| column | definition |
|---|---|
| `label` | Class label as used in InsectSet459 and in the model (genus_epithet). |
| `scientificName` | Species name as used in InsectSet459 (binomial, no authorship). |
| `order` | Order, from the COL24.4 classification of the accepted taxon (from GBIF where the name is absent from COL24.4). |
| `family` | Family, from the same source as order. |
| `col_taxonID` | Identifier of the accepted taxon in Catalogue of Life release COL24.4; empty if the name is absent from COL24.4. |
| `col_acceptedNameUsage` | Accepted name in COL24.4, with authorship. COL24.4 often includes the subgenus. |
| `col_taxonomicStatus` | Status of scientificName in COL24.4: 'accepted' (including names that differ from the accepted name only by the subgenus) or 'synonym'; empty if absent. |
| `gbif_taxonKey` | Key of the matching name in the GBIF Backbone Taxonomy (exact match for every row). |
| `gbif_taxonomicStatus` | Status of that name in the GBIF backbone ('accepted' or 'doubtful'). |
| `itis_tsn` | ITIS taxonomic serial number of scientificName; empty if ITIS does not list the name. |
| `itis_taxonomicStatus` | ITIS usage of that TSN ('valid' or 'invalid'); empty if ITIS does not list the name. |
| `itis_acceptedTsn` | Accepted TSN in ITIS where itis_taxonomicStatus is 'invalid'; empty otherwise. |

## Sources

- Species list: InsectSet459 v1.0 annotation file (Zenodo, doi:10.5281/zenodo.15675220; MD5 a285171a0210fef24b2e25ee08d83213).
- Catalogue of Life Checklist, release COL24.4 (2024-04-26; ChecklistBank dataset 294826; doi:10.48580/dg4lg), the release cited by the InsectSet459 authors for the Orthoptera nomenclature.
- GBIF Backbone Taxonomy, via the GBIF species-matching API (queried 2026-10-08).
- Integrated Taxonomic Information System (ITIS) web service (queried 2026-10-08).

## Coverage and name status

- Catalogue of Life: 449 of 451 species. *Conocephalus fuscus* and *Euthystira brachyptera* have no COL24.4 entry; their order and family are taken from GBIF.
- GBIF: 451 of 451 (exact name matches). Four names have status 'doubtful': *Cicada barbara*, *C. cretensis*, *C. mordoganensis* and *C. orni*.
- ITIS: 188 of 451. Four names are 'invalid': *Diceroprocta grossa*, *Gryllodes sigillatus*, *Okanagana canadensis* and *O. hesperia*.
- COL24.4 treats 4 names as synonyms (`col_taxonomicStatus` = synonym): *Chorthippus bornhalmi*, *Diceroprocta grossa*, *Okanagana canadensis* and *O. hesperia*. `col_acceptedNameUsage` gives the accepted name.
- Class labels are kept exactly as in InsectSet459; no names were changed or merged. No two classes share the same accepted name in COL24.4 (ignoring subgenus).
