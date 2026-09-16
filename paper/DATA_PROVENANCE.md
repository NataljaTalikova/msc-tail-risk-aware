# Data provenance and redistribution record

## Release boundary

Version 1.0 does not redistribute third-party raw data, prepared row-level
panels, or row-level model predictions. The public records contain code,
aggregate exact result tables, metric parameters, figures, and verification
outputs. This conservative boundary avoids applying the project's MIT or CC BY
licences to data supplied under third-party terms.

The local timestamps below identify the frozen files used by the workflow; they
are filesystem snapshot timestamps and are not asserted as independently
verified retrieval dates. SHA-256 fingerprints allow the author to identify
the exact local inputs later. A fresh download can differ from the frozen
snapshot because providers may revise historical observations.

## Sources

This registry was cross-checked against the dissertation's “Reproducibility”
section and “Appendix A: Data Vocabulary” (dissertation pages 38–40). The
series identifiers in the local files and notebook resolve line-wrapped labels
in the dissertation table, including `DCOILBRENTEU`, `XUDLUSS`, and the Yahoo
Finance symbols beginning with `^`.

| Series | Local input | Provider/source recorded in the dissertation | Coverage or use |
|---|---|---|---|
| BSIF.L, HTG.L, SSE.L, TLW.L, UKW.L OHLCV | `daily_data/*_daily.csv` | Yahoo Finance through the `yfinance` API | Daily security prices and volumes |
| FTSE 100 (`^FTSE`), FTSE 250 (`^FTMC`), FTSE All-Share (`^FTAS`) | `daily_data/^FTSE_daily.csv`, `^FTMC_daily.csv`, `^FTAS_daily.csv` | Yahoo Finance through the `yfinance` API | Daily market-index series |
| Bank Rate | `not_ready_or_daily_data/Bank_Rate_history_BoE_start_from_merge.csv` | [Bank of England Bank Rate database](https://www.bankofengland.co.uk/boeapps/database/Bank-Rate.asp) | Policy-rate step series |
| SONIA | `not_ready_or_daily_data/SONIA.csv` | [Bank of England SONIA benchmark](https://www.bankofengland.co.uk/markets/sonia-benchmark) | Daily short-rate series |
| GBP/USD spot rate, Bank of England series `XUDLUSS` | `not_ready_or_daily_data/GBP_USD_Spot_rate_1_Jan_2015_to_18_July_2025.csv` | Bank of England; Appendix A identifies the provider and the local source file identifies series `XUDLUSS` | Conversion of Brent USD price to GBP |
| CPIH | `not_ready_or_daily_data/CPIH_monthly_Oct_2013_to_June_2025.xlsx` | [UK Office for National Statistics consumer price indices dataset](https://www.ons.gov.uk/economy/inflationandpriceindices/datasets/consumerpriceindices) | Monthly release-aligned inflation series |
| Brent Europe spot price, FRED series `DCOILBRENTEU` | `not_ready_or_daily_data/DCOILBRENTEU_1 Jan_2015_to_18_July_2025.csv` | [Federal Reserve Economic Data](https://fred.stlouisfed.org/graph/?g=1BRV) | Daily Brent price in USD |
| VIX, FRED series `VIXCLS` | `not_ready_or_daily_data/VIX.csv` | [Federal Reserve Economic Data](https://fred.stlouisfed.org/series/VIXCLS) | Daily CBOE volatility index |
| OVX, FRED series `OVXCLS` | `not_ready_or_daily_data/OVX.csv` | [Federal Reserve Economic Data](https://fred.stlouisfed.org/series/OVXCLS) | Daily CBOE oil-volatility index |
| UK 10-year gilt yield (dissertation label `GUKG10`; project field `GUK10Y`) | `not_ready_or_daily_data/GUK10Y.csv` | [Investing.com UK 10-year bond-yield history](https://uk.investing.com/rates-bonds/uk-10-year-bond-yield-historical-data) | Daily long-rate proxy |
| BSIF shares outstanding | `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | Bluefield Solar Income Fund factsheets, Q1 2015–Q1 2025, from [bluefieldsif.com](https://bluefieldsif.com) | Sparse shares-outstanding anchors used to construct market-capitalisation features |
| SSE shares/market-cap history | `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | [CompaniesMarketCap: SSE](https://companiesmarketcap.com/gbp/sse/marketcap/) | Sparse anchors used to construct market-capitalisation features |
| Hunting shares/market-cap history | `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | [CompaniesMarketCap: Hunting](https://companiesmarketcap.com/gbp/hunting-plc/marketcap/) | Sparse anchors used to construct market-capitalisation features |
| Tullow Oil shares/market-cap history | `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | [CompaniesMarketCap: Tullow Oil](https://companiesmarketcap.com/tullow-oil/marketcap/) | Sparse anchors used to construct market-capitalisation features |
| Greencoat UK Wind shares/market-cap history | `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | [CompaniesMarketCap: Greencoat UK Wind](https://companiesmarketcap.com/gbp/greencoat-uk-wind/marketcap/) | Sparse anchors used to construct market-capitalisation features |

The dissertation identifies a provider for every external input used by the
analysis. It does not record an exact Bank of England URL for `XUDLUSS`, exact
retrieval dates for every download, or third-party redistribution licences.
Those details therefore are not inferred here. The live Yahoo Finance refresh
cell remains disabled in `EDA.ipynb`; the article run uses the frozen local CSV
snapshots recorded below.

## Frozen-input fingerprints

| Local file | Bytes | Local snapshot timestamp | SHA-256 |
|---|---:|---|---|
| `daily_data/^FTAS_daily.csv` | 227656 | 2025-09-12 00:02:41 +01:00 | `24d769199ebfa3477f51a59dece7abc87ddcf9c9ba5850343d36ed008e2994c9` |
| `daily_data/^FTMC_daily.csv` | 208044 | 2025-09-12 00:02:40 +01:00 | `495b61ec6bf920592a5f71fa2a3249d81959f2149845743f3aeb0efb7cfb0933` |
| `daily_data/^FTSE_daily.csv` | 213761 | 2025-09-12 00:02:40 +01:00 | `f33d5db9c839d93ba938a66c57aaae31f58d2afc79d14b6406aa5cfea35175a5` |
| `daily_data/BSIF_daily.csv` | 249483 | 2025-09-12 00:02:38 +01:00 | `b41a108778325326c01e8dcd178d1940324e24bddd2a2e1f1db333a55500e867` |
| `daily_data/HTG_daily.csv` | 242186 | 2025-09-12 00:02:39 +01:00 | `18086333377925eeda91ea331b8bdb4f316f984730497913fcf3f47fd756fcbc` |
| `daily_data/SSE_daily.csv` | 250963 | 2025-09-12 00:02:40 +01:00 | `595ef8af4eb367f41bebc1e8ca420cbdd5ca55c6011f65d2a23e186a81199d25` |
| `daily_data/TLW_daily.csv` | 239321 | 2025-09-12 00:02:39 +01:00 | `d3a405d86010b9428dacfdd00b519a767adf16d1c22b36486afc6ea1f06fd412` |
| `daily_data/UKW_daily.csv` | 251854 | 2025-09-12 00:02:38 +01:00 | `8723e5d1e597edd054ae2d48655b7bd46e53b040e270925534a3f7104233ad14` |
| `not_ready_or_daily_data/Bank_Rate_history_BoE_start_from_merge.csv` | 434 | 2025-08-16 15:02:53 +01:00 | `6646921b5e8dbce293daf5952b5667aa5235fe8a0a9f336763fd71b6b01a9d8d` |
| `not_ready_or_daily_data/CPIH_monthly_Oct_2013_to_June_2025.xlsx` | 15834 | 2025-08-16 17:55:34 +01:00 | `431a2d878d22bbe67d50803fa2417f0d4ac8d4be101df3e2dccc23032622cb2d` |
| `not_ready_or_daily_data/DCOILBRENTEU_1 Jan_2015_to_18_July_2025.csv` | 48979 | 2025-07-27 23:00:42 +01:00 | `cb5252df7a2cb2a456bd0e2123f3af1c6679fdda21a4b323b07cb9ba2c30357a` |
| `not_ready_or_daily_data/GBP_USD_Spot_rate_1_Jan_2015_to_18_July_2025.csv` | 47790 | 2025-07-28 00:09:31 +01:00 | `9db0aec8ed7b7e8c80a4852e90aaeb9eb3964aeb63f79ef265d1768ff7db74b2` |
| `not_ready_or_daily_data/GUK10Y.csv` | 57147 | 2025-08-02 16:14:59 +01:00 | `3b438d43d0508f562bfbaf66fc0c351bf7a11495ddd4d4e8e84d1809d53c37c6` |
| `not_ready_or_daily_data/Market_Cap_in_billions.xlsx` | 10175 | 2025-07-31 21:38:06 +01:00 | `008f863ae481fdca5be434f6af1b5f1506ecefbeec3a693ddbbdd64e92e56e1d` |
| `not_ready_or_daily_data/OVX.csv` | 48763 | 2025-08-02 16:04:32 +01:00 | `45d05c1a685c81f6ac181cf0bdb18b9df32db0a9d793434eccd95e8e6b73cd79` |
| `not_ready_or_daily_data/SONIA.csv` | 47171 | 2025-08-02 16:05:05 +01:00 | `67e1ab233817a045272226757d96088e10235e5bc6f5de73f60c18380a15eff8` |
| `not_ready_or_daily_data/VIX.csv` | 48811 | 2025-08-02 15:32:18 +01:00 | `66fea7f266b5d84b7b799cd2007753847428e51ecd9b7b469971f3c91627790e` |

## Public release contents derived from the inputs

The following compact artifacts are released because they report aggregate
calculations rather than redistributing the source time series:

- `common_panel_metrics.csv`;
- `article_model_summary.csv`;
- `metric_parameters.csv`;
- `verification_summary.json`;
- aggregate figures and their compact plotting tables.

The excluded files remain available in the author's frozen local research
archive. Reproduction from newly retrieved inputs should be treated as a fresh
replication because vendor revisions may change individual observations.
