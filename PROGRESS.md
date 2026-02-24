# NFL Analytics プロジェクト進捗

## 目的
Dallas Cowboys Strategic Football Fellow 応募（締切: 2026/5/13）に向けて、レジュメに記載済みのスキル（nflfastR, NGS, EPA, CPOE, トラッキングデータ）を裏付けるNFL分析ポートフォリオを作成する。

## 方針決定の経緯

### 分析テーマの選定
- 元プラン（COWBOYS_PLAN.md）には EPA分析、レシーバーセパレーション、ブリッツ予測、D&D傾向の4候補があった
- 「まず使えるデータから逆算して実用的な分析を設計しよう」という方針に転換
- nfl_data_py で取得可能なデータ（PBP 397カラム、NGS、Snap Counts等）を洗い出し
- 6つの候補（スカウティングレポート、4th Down、パスラッシュ、人員パッケージ、WRセパレーション、QBプレッシャー）から絞り込み
- **人員パッケージ効率分析**を第1弾に決定（SFOの実務直結）
- 座標データ（Big Data Bowl）の存在を確認 → **個人マッチアップ分析**を第2弾に決定（募集要項の「Zebra トラッキング技術」に直結）

### データソースの選定
- BDB 2025: ホストがデータ削除済みで取得不可
- BDB 2024: APIからデータアクセス不可
- **BDB 2023 を採用**: 2021シーズン W1-8、920MB、PFFスカウティングデータ付き

## 環境構成
- **実行環境**: 既存Dockerコンテナ `accounting_worker`（farm-tools イメージ）
- **追加インストール**: nfl_data_py, jupyter, scikit-learn, seaborn, xgboost, kagglehub
- **データ取得**: Kaggle API（Bearer トークン認証）で BDB 2023 全ファイルをダウンロード
- **リポジトリ**: `C:\Users\shunn\git\nfl-analytics\`（git init 済み、GitHub push は未実施）

## 完成した成果物

### Notebook 1: Personnel Package Efficiency Analysis
- **ファイル**: `notebooks/01_personnel_efficiency.ipynb`
- **データ**: nflfastR 2024シーズン PBP データ（33,336プレー）
- **内容**:
  1. 人員パッケージのパース（生データ → 標準表記: 11, 12, 21等）
  2. オフェンス人員別 EPA/play ランキング（22 personnelが最高EPA +0.063）
  3. パス率 vs EPA の散布図
  4. **人員マッチアップマトリクス**（オフェンス × ディフェンスのEPAヒートマップ）
  5. ディフェンスの対応パターン（ボックス内人数、パスラッシャー数）
  6. **Cowboys 専用分析**（使用率 vs リーグ平均、人員別EPA）
  7. フォーメーション × 人員のインタラクション
  8. **Personnel Predictability Index**（人員からプレーコールが読まれやすいチームのランキング）
- **主な発見**:
  - Cowboys は 11 personnel に偏重（71% vs リーグ63%）
  - Cowboys の Predictability Index はリーグ上位（プレーコールが読まれやすい）
  - 22 personnel vs 2-4-5 defense で +0.284 EPA（最も有利なマッチアップ）

### Notebook 2: Pass Rush Win Rate & Individual Matchup Analysis
- **ファイル**: `notebooks/02_pass_rush_matchups.ipynb`
- **データ**: BDB 2023 トラッキングデータ（W1-4, 435万行）+ PFF スカウティングデータ（188,254レコード）
- **内容**:
  1. Pass Rush Win Rate（PRWR）ランキング（148人の有資格ラッシャー）
  2. **個人マッチアップマトリクス**（OL vs DL の勝率ヒートマップ、PFFブロッキングアサインメント使用）
  3. **Time-to-Pressure 分析**（トラッキング座標から QB までの距離カーブを計算）
  4. 個人ラッシャーのクロージングスピードプロファイル（上位6人）
  5. プレッシャーがプレー結果に与える影響（Comp% 65.1% → 35.1%）
  6. カバレッジタイプ別プレッシャー率
- **主な発見**:
  - Myles Garrett が PRWR 24.0% でトップ（208ラッシュ、12サック）
  - プレッシャー生成プレーと非生成プレーで、スナップ後1.5秒から距離カーブが明確に分岐
  - 被プレッシャー時は平均獲得ヤードが 8.1 → 3.9 に激減

## リポジトリ構成（現状）
```
nfl-analytics/
├── .git/
├── .gitignore
├── requirements.txt
├── PROGRESS.md
├── notebooks/
│   ├── 01_personnel_efficiency.ipynb   ← 完成（出力付き）
│   ├── 02_pass_rush_matchups.ipynb     ← 完成（出力付き）
│   ├── 03_motion_coverage.ipynb        ← 完成（出力付き）
│   ├── 04_shell_classification.ipynb   ← 完成（出力付き）
│   ├── build_01_personnel.py           ← notebook生成スクリプト
│   ├── build_02_tracking.py            ← notebook生成スクリプト
│   ├── build_03_motion_coverage.py     ← notebook生成スクリプト
│   ├── build_04_shell_classification.py ← notebook生成スクリプト
│   ├── build_01_epa.py                 ← 旧版（未使用）
│   └── *.png                           ← 生成されたチャート画像
├── data/
│   └── bdb2023/                        ← BDB 2023 データ（.gitignore済み）
│       ├── games.csv
│       ├── players.csv
│       ├── plays.csv
│       ├── pffScoutingData.csv
│       └── week1.csv ~ week8.csv
└── src/                                ← 未使用
```

### Notebook 3: Motion-Based Coverage Classification
- **ファイル**: `notebooks/03_motion_coverage.ipynb`（実行済み・出力付き）
- **データ**: BDB 2023 トラッキングデータ（W1-8, 全8週 ~870万行）+ PFF スカウティングデータ + plays.csv（カバレッジラベル）
- **内容**:
  1. プリスナップモーション検出（`man_in_motion`/`shift` イベント + 速度ベース検出で **1,422プレー** を特定）
  2. カバレッジラベル準備（PFF `pff_passCoverageType`: Man 378 / Zone 847）
  3. **DB Follow Score** — モーション選手と最近傍DBの横方向変位のPearson相関で追従度を数値化（**1,203プレー**で特徴量計算成功）
  4. 特徴量エンジニアリング（follow_score, lateral_mirror, reaction_delay, separation_delta, handoff_flag）
  5. 探索的可視化（Man vs Zone の特徴量分布、散布図、サンプル軌跡）
  6. カバレッジ分類モデル（XGBoost: ベースライン vs フルモデル比較）
  7. **SHAP分析** — feature importance + Beeswarm plot
  8. マッチアップゾーン調査 — PCA + 誤分類分析でハイブリッドカバレッジの特性を分析
- **モデル性能**:
  - ベースライン（文脈特徴量のみ）: AUC 0.746, Accuracy 0.744, F1 macro 0.663
  - **フルモデル（+ モーション反応特徴量）: AUC 0.820, Accuracy 0.774, F1 macro 0.731**
  - モーション反応特徴量の追加で AUC +0.074 改善
  - Man: precision 0.64 / recall 0.60、Zone: precision 0.83 / recall 0.85
- **主な発見**:
  - Zoneの方がManよりfollow_scoreが高い（0.545 vs 0.422）— Zone DBもモーションに反応してシフトするため
  - lateral_mirrorでManの方が高い（0.375 vs 0.276）— ManDBはモーション選手と同等距離を移動
  - 「Man的なZoneプレー」の14.9%はCover-3が多く（19/31件）、マッチアップゾーン的な挙動を示唆
  - PCA分析: PC1（分散59.9%）はseparation_deltaが支配、PC2（22.3%）はfollow_scoreが支配
- **差別化**: BDB 2025ファイナリスト（NN, HMM, Transformer）と比較し、QBの実際のプリスナップリードを数値化する**解釈可能なアプローチ**
- **生成スクリプト**: `notebooks/build_03_motion_coverage.py`
- **追加依存**: `shap>=0.42.0`, `scipy>=1.10.0`

### Notebook 4: Pre-Snap Coverage Shell Classification
- **ファイル**: `notebooks/04_shell_classification.ipynb`（実行済み・出力付き）
- **データ**: BDB 2023 トラッキングデータ（W1-8）+ PFF スカウティングデータ（セーフティアラインメント）+ plays.csv（カバレッジラベル）
- **内容**:
  1. カバレッジシェルラベル作成（Cover-0/1/3 → 1-High 4,946件、Cover-2/Quarters/6/2-Man → 2-High 3,123件）
  2. セーフティ位置抽出（PFF `pff_positionLinedUp`: FS/FSL/FSR/SS/SSL/SSR）+ play direction正規化
  3. **ルールベースシェル分類器**（閾値グリッドサーチ: 深さ × 横方向距離）
  4. **カバレッジディスガイズ検出**（プリスナップシェル ≠ 実際のカバレッジ）
  5. **チーム別ディスガイズ傾向**（32チームランキング）
  6. ディスガイズのオフェンスへの影響（獲得ヤード、完了率、INT率）
  7. セーフティローテーション検出（ポストスナップ1.5秒の移動追跡）
  8. **QB意思決定ツリー**（Shell × Man/Zone → 具体的カバレッジの推定）
- **分類性能**:
  - ルールベース分類器: **Accuracy 0.744**（depth ≥ 5 yd, lateral_sep ≥ 12 yd）
  - 1-High: precision 0.75 / recall 0.85 / F1 0.80
  - 2-High: precision 0.72 / recall 0.58 / F1 0.64
- **主な発見**:
  - **ディスガイズ率: 25.6%**（4分の1のプレーでプリスナップシェルと実際のカバレッジが不一致）
  - Cover-2が最もディスガイズされやすい（48.5%）、次いでQuarters（41.9%）、2-Man（41.4%）
  - チーム別ディスガイズ率（2021 W1-8）:
    - **Top 5 (高ディスガイズ)**: ATL 39.3%, TB 38.6%, NYG 35.9%, CLE 35.3%, LA 34.9%
    - **Bottom 5 (ほぼ正直)**: HOU 8.2%, LV 15.1%, MIN 15.9%, NE 16.3%, NYJ 18.4%
    - DAL はリーグ中位付近
  - ディスガイズのオフェンス結果への影響は**統計的に有意でない**（p = 0.67）— ディスガイズ自体は成功を保証しない
  - ダウン別: 2nd downではディスガイズ時 -0.33 yd/play の効果あり、3rd downでは逆に +0.10 yd（パッシングダウンではディスガイズ効果薄い）
  - Shell + Man/Zone の2軸で具体的カバレッジを推定: **75.5% accuracy**（1-High+Zone → Cover-3 が100%的中）
  - 2-High+Zone のみ37.4%（Cover-2/Quarters/Cover-6 の3択で曖昧）
  - QB意思決定ツリーの精度: 1-High+Man → Cover-1 (94.9%), 1-High+Zone → Cover-3 (100%), 2-High+Man → 2-Man (100%)
  - セーフティローテーション: ディスガイズプレーの24.1%でDeep Safety が3+yd LOS方向に移動（honest play は6.4%）
- **生成スクリプト**: `notebooks/build_04_shell_classification.py`

## 未実施タスク
- [ ] README.md 作成
- [ ] GitHub パブリックリポジトリ作成 & push（gh CLI 未インストール）
- [ ] 追加 notebook の検討（4th Down 分析、スカウティングレポート自動生成 等）
- [ ] Tableau ダッシュボード（COWBOYS_PLAN.md の ToDo #2）
- [ ] カバーレター（COWBOYS_PLAN.md の ToDo #3）
- [ ] ポートフォリオ更新（COWBOYS_PLAN.md の ToDo #5）
- [ ] 応募提出（COWBOYS_PLAN.md の ToDo #6）

## 技術メモ
- Docker コンテナ `accounting_worker` は `farm` → `/app` マウント。`nfl-analytics` は `/app` 外なので `docker cp` でファイル転送が必要
- Kaggle API は新形式トークン（KGAT_xxx）を Bearer 認証で使用。kaggle CLI は新トークン非対応のため REST API を直接使用
- BDB データファイルは Kaggle が zip 形式で配信するため、`io.BytesIO` + `zipfile` でインメモリ解凍が必要
- notebook は nbformat で Python スクリプトから生成 → `jupyter nbconvert --execute` で出力付き実行
