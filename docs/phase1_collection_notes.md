# RAGBench — Phase 1 Documentation Collection Notes

**Project:** RAGBench — RAG Evaluation Framework  
**Phase:** 1 — Documentation Collection & Preparation  
**Date:** November 2026  
**Status:** Implemented, Audited, and Verified  

---

## 1. Overview and Objectives

Phase 1 establishes the foundational technical knowledge base for the RAGBench evaluation framework. The objective is to collect, inspect, structure, and validate official documentation from five distinct technology sources:

1. **Python Documentation** (Language concepts, standard library, data structures)
2. **Scikit-learn Documentation** (Machine learning algorithms, preprocessing, pipelines, evaluation)
3. **Docker Documentation** (Containers, images, Compose, networking, storage, Dockerfiles)
4. **FastAPI Documentation** (APIs, path/query parameters, dependencies, security, testing)
5. **AWS Documentation** (Cloud architecture across IAM, S3, EC2, and Lambda)

**Key Principles:**
- **Zero Chunking:** Documents are preserved as complete single pages (1 documentation page = 1 document). Chunking is strictly deferred to Phase 2.
- **Strict Provenance:** Every document retains source repository, branch/commit SHA, official URL, license, and content SHA-256 hash.
- **Content Integrity:** Code examples, tables, warning/note blocks, and heading hierarchies are preserved.
- **Deterministic Processing:** Running `scripts/prepare_docs.py` repeatedly produces identical output.

---

## 2. Documentation Sources & Provenance Audit

### Source 1: Python
- **Official Website:** `https://docs.python.org/3/`
- **Source Repository:** `https://github.com/python/cpython`
- **Branch:** `main`
- **Content Source Revision:** `unknown` (Files were fetched via HTTP from the raw endpoint of the `main` branch without pinning an immutable commit SHA at fetch time).
- **License Verification Revision:** `0d3ab6f17e1119095ce9c452cd14c15e60256b50` (Inspected repository root `LICENSE` file on `main`).
- **Verification Status:** `unknown` for exact content collection commit; verified for repository license.
- **Version Clarification:** CPython `main` branch snapshot (tracks in-development Python 3.14 tree; the collected tutorial files from `Doc/tutorial/` cover foundational language syntax and features identical to current stable Python 3.12/3.13 documentation at `docs.python.org/3/`).
- **License:** Python Software Foundation License Version 2 (PSF-2.0)
- **Local Raw Directory:** `data/raw/python/`
- **Pages Collected (12 pages):**
  - `appetite.rst`: Whetting Your Appetite (Python tutorial intro)
  - `interpreter.rst`: Using the Python Interpreter
  - `introduction.rst`: An Informal Introduction to Python (numbers, strings, lists)
  - `controlflow.rst`: More Control Flow Tools (if, for, range, functions)
  - `datastructures.rst`: Data Structures (list comprehensions, tuples, sets, dicts)
  - `modules.rst`: Modules and Packages
  - `inputoutput.rst`: Input and Output (file reading/writing, JSON)
  - `errors.rst`: Errors and Exceptions
  - `classes.rst`: Classes and Object-Oriented Programming
  - `stdlib.rst`: Brief Tour of the Standard Library
  - `stdlib2.rst`: Brief Tour of the Standard Library — Part II
  - `venv.rst`: Virtual Environments and Packages

### Source 2: Scikit-learn
- **Official Website:** `https://scikit-learn.org/stable/`
- **Source Repository:** `https://github.com/scikit-learn/scikit-learn`
- **Branch:** `main`
- **Content Source Revision:** `unknown` (Files were fetched via HTTP from the raw endpoint of the `main` branch without pinning an immutable commit SHA at fetch time).
- **License Verification Revision:** `d5a61d1149cb94a936d4b32212330f4ba8f24ce0` (Inspected repository `COPYING` file on `main`).
- **Verification Status:** `unknown` for exact content collection commit; verified for repository license.
- **Version Clarification:** Scikit-learn `main` branch snapshot (tracks in-development releases; the collected user guide chapters from `doc/` match the concepts, algorithms, and metrics documented in the current stable User Guide at `scikit-learn.org/stable/`).
- **License:** BSD 3-Clause "New" or "Revised" License
- **Local Raw Directory:** `data/raw/scikit_learn/`
- **Pages Collected (14 pages):**
  - `getting_started.rst`: Getting Started with Scikit-learn
  - `supervised_learning.rst`: Supervised Learning Overview
  - `unsupervised_learning.rst`: Unsupervised Learning Overview
  - `data_transforms.rst`: Dataset Transformations Overview
  - `model_selection.rst`: Model Selection and Evaluation Overview
  - `common_pitfalls.rst`: Common Pitfalls and Recommended Practices
  - `modules_preprocessing.rst`: Data Preprocessing and Feature Scaling
  - `modules_feature_selection.rst`: Feature Selection
  - `modules_cross_validation.rst`: Cross-Validation: Evaluating Estimator Performance
  - `modules_grid_search.rst`: Tuning Hyper-parameters of an Estimator
  - `modules_model_evaluation.rst`: Metrics and Scoring: Quantifying Prediction Quality
  - `modules_compose.rst`: Pipelines and Composite Estimators
  - `modules_impute.rst`: Imputation of Missing Values
  - `modules_clustering.rst`: Clustering Algorithms

### Source 3: Docker
- **Official Website:** `https://docs.docker.com/`
- **Source Repository:** `https://github.com/docker/docs`
- **Branch:** `main`
- **Content Source Revision:** `unknown` (Files were fetched via HTTP from the `main` branch without pinning an immutable commit SHA at fetch time. The earlier candidate SHA `a342371cfda92b772c57a9099859f7b19cecd315` was verified locally via Git against `docker/docs.git` and confirmed NOT to be a valid object/ref).
- **License Verification Revision:** `4937e25aa671f382c793fd03ba457bdb25e0d028` (Inspected repository root `LICENSE` file via Git clone at HEAD).
- **Verification Status:** `unknown` for original content collection commit; `verified` for repository license at the inspected revision.
- **Version Clarification:** Docker Docs `main` branch snapshot (corresponds to current official documentation published at `docs.docker.com`).
- **License & Verification Evidence:** Apache License 2.0 (Apache-2.0). Inspected repository root `LICENSE` file via Git clone at verified commit `4937e25aa671f382c793fd03ba457bdb25e0d028`. The file contains the full Apache License Version 2.0 text (Copyright 2016 Docker, Inc.). Under Section 1 ("Definitions"), "Source" form explicitly includes "documentation source, and configuration files", and Section 2 grants copyright license for the Work across both documentation source files (`.md`) and code examples without distinction.
- **Local Raw Directory:** `data/raw/docker/`
- **Pages Collected (13 pages):**
  - `docker-overview.md`: Docker Architecture & Overview
  - `what-is-a-container.md`: What is a Container?
  - `what-is-an-image.md`: What is an Image?
  - `what-is-docker-compose.md`: What is Docker Compose?
  - `what-is-a-registry.md`: What is a Container Registry?
  - `writing-a-dockerfile.md`: Writing a Dockerfile
  - `build-tag-and-publish-an-image.md`: Build, Tag, and Publish an Image
  - `multi-stage-builds.md`: Multi-stage Builds
  - `using-the-build-cache.md`: Using the Build Cache
  - `publishing-ports.md`: Publishing Ports (Networking)
  - `persisting-container-data.md`: Persisting Container Data (Volumes)
  - `sharing-local-files.md`: Sharing Local Files (Bind Mounts)
  - `multi-container-applications.md`: Multi-container Applications

### Source 4: FastAPI
- **Official Website:** `https://fastapi.tiangolo.com/`
- **Source Repository:** `https://github.com/fastapi/fastapi`
- **Branch:** `master`
- **Content Source Revision:** `unknown` (Files were fetched via HTTP from the raw endpoint of the `master` branch without pinning an immutable commit SHA at fetch time).
- **License Verification Revision:** `80d475b44a278253d15744a683af9614853af19e` (Inspected repository root `LICENSE` file on `master`).
- **Verification Status:** `unknown` for exact content collection commit; verified for repository license.
- **Version Clarification:** FastAPI `master` branch snapshot (tracks FastAPI 0.115+; the `master` branch directly drives the deployed official documentation at `fastapi.tiangolo.com`).
- **License:** MIT License
- **Local Raw Directory:** `data/raw/fastapi/`
- **Pages Collected (14 pages):**
  - `index.md`: FastAPI Introduction
  - `tutorial_first-steps.md`: First Steps
  - `tutorial_path-params.md`: Path Parameters
  - `tutorial_query-params.md`: Query Parameters
  - `tutorial_body.md`: Request Body
  - `tutorial_response-model.md`: Response Model - Return Type
  - `tutorial_dependencies_index.md`: Dependencies - First Steps
  - `tutorial_security_index.md`: Security - First Steps
  - `tutorial_middleware.md`: Middleware
  - `tutorial_handling-errors.md`: Handling Errors
  - `tutorial_background-tasks.md`: Background Tasks
  - `tutorial_testing.md`: Testing
  - `tutorial_sql-databases.md`: SQL (Relational) Databases
  - `deployment_index.md`: Deployment Concepts

### Source 5: AWS
- **Official Website:** `https://docs.aws.amazon.com/`
- **Source Repository:** N/A (Official HTML rendered documentation pages used because public Markdown documentation trees are not maintained on GitHub for all four services)
- **Content Source Revision:** `N/A` (Captured directly from published official HTML documentation endpoints).
- **License Verification Revision:** `N/A`
- **Verification Status:** `verified` (Captured from live official AWS endpoints).
- **Version:** AWS Documentation Current (November 2026)
- **License:** Documentation content: Creative Commons Attribution-ShareAlike 4.0 International (CC-BY-SA-4.0); Sample code and code examples: MIT No Attribution (MIT-0). Reference: AWS Open Source Documentation and Code Samples licensing terms (`https://aws.amazon.com/terms/`).
- **Collection Method & Site Terms Considerations:**
  - *Documented Technical Method:* Targeted, rate-limited HTTP retrieval of 12 specific conceptual HTML pages across IAM, S3, EC2, and Lambda with an identified User-Agent (`RAGBench-Collector/1.0`) and inter-request delay intervals designed to minimize server impact.
  - *Important Legal Distinction:* This technical design was chosen for reproducibility and to prevent placing unreasonable load on AWS infrastructure. This technical practice is distinct from a legal determination that the collection complied with AWS Site Terms (e.g. Section 3 "Site Access and Content"). RAGBench makes no claim of explicit authorization, license waiver, or guaranteed legal compliance under AWS Site Terms. Users are responsible for evaluating their own compliance.
- **Local Raw Directory:** `data/raw/aws/`
- **Pages Collected (12 pages: 3 per service):**
  - **IAM:**
    - `iam_introduction.html`: What is IAM?
    - `iam_identities.html`: IAM Identities (Users, User Groups, and Roles)
    - `iam_policies.html`: Policies and Permissions in IAM
  - **Amazon S3:**
    - `s3_welcome.html`: What is Amazon S3?
    - `s3_buckets_objects.html`: Working with Amazon S3 Buckets
    - `s3_access_control.html`: Identity and Access Management for Amazon S3
  - **Amazon EC2:**
    - `ec2_concepts.html`: What is Amazon EC2?
    - `ec2_instances.html`: Amazon EC2 Instances
    - `ec2_security_groups.html`: Amazon EC2 Security Groups
  - **AWS Lambda:**
    - `lambda_welcome.html`: What is AWS Lambda?
    - `lambda_runtime_environment.html`: AWS Lambda Execution Environment
    - `lambda_execution_role.html`: AWS Lambda Execution Role

---

## 3. Processing and Normalization Strategy

`scripts/prepare_docs.py` processes raw files without network dependencies:

1. **RST Parsing:** Converts Sphinx and docutils reStructuredText into clean Markdown. Converts headings (`=`, `-`, `~`), code directives (`.. code-block::`), inline code (`` `code` ``), and admonitions (`.. note::`, `.. warning::`) into GitHub-flavored Markdown.
2. **Markdown Parsing:** Normalizes Hugo/MkDocs Markdown, extracts metadata/frontmatter titles, sanitizes MkDocs anchor tags (`{#anchor}`), strips Hugo shortcodes (`{{< ... >}}`), and standardizes heading structures.
3. **HTML Parsing:** Uses standard library `html.parser.HTMLParser` targeting the article container (`main-col-body` in AWS docs) to discard navigation menus, breadcrumbs, and scripts while preserving text paragraphs, lists, code samples, tables, and heading hierarchies.
4. **Content Hashing:** Computes SHA-256 over normalized `content_markdown` to guarantee tamper-proof and verifiable data tracking.
5. **Deterministic JSONL:** All raw files are sorted alphabetically prior to processing so repeated runs yield identical outputs.

---

## 4. Git Repository Status

- **Working Tree State:** All project files and directories are **untracked locally** (`??` in `git status`).
- **Commits:** **0 commits** have been made.
- **Remotes:** **No remote is configured** (no remote repository, no push performed).
- **Ignored Paths:** `.venv/`, `.env`, and cache directories (`.pytest_cache/`, `.ruff_cache/`) are strictly excluded via `.gitignore`.
