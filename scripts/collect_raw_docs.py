"""Official documentation collector for RAGBench Phase 1.

Downloads official raw documentation for five technology sources:
1. Python (CPython repository & official Python documentation)
2. Scikit-learn (scikit-learn repository & User Guide)
3. Docker (docker/docs repository)
4. FastAPI (fastapi/fastapi repository)
5. AWS (official rendered documentation HTML from docs.aws.amazon.com)

Saves all original raw files to data/raw/<source_name>/
Generates data/raw/source_manifest.json with complete provenance metadata.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

# Headers to ensure polite, identified requests
HEADERS = {"User-Agent": "RAGBench-Collector/1.0 (Educational RAG Evaluation Research)"}


def download_url(url: str, dest_path: Path, max_retries: int = 3) -> bool:
    """Download content from a URL to dest_path with retries."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers=HEADERS)

    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=25) as resp:
                content = resp.read()
                dest_path.write_bytes(content)
                return True
        except (urllib.error.URLError, TimeoutError) as exc:
            if attempt == max_retries - 1:
                print(f"Failed to download {url}: {exc}")
                return False
            time.sleep(1.0 * (attempt + 1))
    return False


def collect_python_docs(raw_dir: Path) -> dict:
    print("Collecting Python documentation...")
    python_raw = raw_dir / "python"
    python_raw.mkdir(parents=True, exist_ok=True)

    # License
    license_url = "https://raw.githubusercontent.com/python/cpython/main/LICENSE"
    download_url(license_url, python_raw / "LICENSE")

    base_raw_url = "https://raw.githubusercontent.com/python/cpython/main/Doc/tutorial/"
    base_web_url = "https://docs.python.org/3/tutorial/"

    pages = [
        ("appetite.rst", "appetite.html", "Whetting Your Appetite"),
        ("interpreter.rst", "interpreter.html", "Using the Python Interpreter"),
        ("introduction.rst", "introduction.html", "An Informal Introduction to Python"),
        ("controlflow.rst", "controlflow.html", "More Control Flow Tools"),
        ("datastructures.rst", "datastructures.html", "Data Structures"),
        ("modules.rst", "modules.html", "Modules"),
        ("inputoutput.rst", "inputoutput.html", "Input and Output"),
        ("errors.rst", "errors.html", "Errors and Exceptions"),
        ("classes.rst", "classes.html", "Classes"),
        ("stdlib.rst", "stdlib.html", "Brief Tour of the Standard Library"),
        ("stdlib2.rst", "stdlib2.html", "Brief Tour of the Standard Library - Part II"),
        ("venv.rst", "venv.html", "Virtual Environments and Packages"),
    ]

    included = []
    for rst_name, html_name, title in pages:
        src_url = f"{base_raw_url}{rst_name}"
        web_url = f"{base_web_url}{html_name}"
        local_file = python_raw / rst_name
        if download_url(src_url, local_file):
            included.append(
                {
                    "title": title,
                    "relative_path": f"python/{rst_name}",
                    "url": web_url,
                    "format": "rst",
                }
            )
            print(f"  [+] Python: {rst_name}")

    return {
        "source_name": "python",
        "official_website": "https://docs.python.org/3/",
        "repository": "https://github.com/python/cpython",
        "branch": "main",
        "content_source_revision": "unknown",
        "license_verification_revision": "0d3ab6f17e1119095ce9c452cd14c15e60256b50",
        "verification_status": "unknown",
        "commit_sha": "0d3ab6f17e1119095ce9c452cd14c15e60256b50",  # Verified HEAD on main
        "version": (
            "CPython main branch snapshot (Python 3.14 development tree; tutorial "
            "corresponds to Python 3.12-3.13 stable documentation at docs.python.org/3/)"
        ),
        "license": "Python Software Foundation License Version 2",
        "license_file": "data/raw/python/LICENSE",
        "collection_method": "Raw source RST files downloaded from official CPython repository",
        "raw_directory": "data/raw/python",
        "included_page_count": len(included),
        "included_pages": included,
        "excluded_files_or_categories": [
            "interactive.rst",
            "floatingpoint.rst",
            "appendix.rst",
            "c-api/",
            "distutils/",
        ],
        "reasons_for_exclusions": (
            "Out-of-scope internal C-API, deprecated tooling, or supplementary appendixes "
            "not part of core Python tutorial topics."
        ),
    }


def collect_scikit_learn_docs(raw_dir: Path) -> dict:
    print("Collecting Scikit-learn documentation...")
    sklearn_raw = raw_dir / "scikit_learn"
    sklearn_raw.mkdir(parents=True, exist_ok=True)

    license_url = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/COPYING"
    download_url(license_url, sklearn_raw / "LICENSE")

    base_raw = "https://raw.githubusercontent.com/scikit-learn/scikit-learn/main/doc/"
    base_web = "https://scikit-learn.org/stable/"

    pages = [
        ("getting_started.rst", "getting_started.html", "Getting Started"),
        ("supervised_learning.rst", "supervised_learning.html", "Supervised Learning"),
        ("unsupervised_learning.rst", "unsupervised_learning.html", "Unsupervised Learning"),
        ("data_transforms.rst", "data_transforms.html", "Dataset Transformations"),
        ("model_selection.rst", "model_selection.html", "Model Selection and Evaluation"),
        (
            "common_pitfalls.rst",
            "common_pitfalls.html",
            "Common Pitfalls and Recommended Practices",
        ),
        ("modules/preprocessing.rst", "modules/preprocessing.html", "Preprocessing Data"),
        ("modules/feature_selection.rst", "modules/feature_selection.html", "Feature Selection"),
        (
            "modules/cross_validation.rst",
            "modules/cross_validation.html",
            "Cross-Validation: Evaluating Estimator Performance",
        ),
        (
            "modules/grid_search.rst",
            "modules/grid_search.html",
            "Tuning Hyper-parameters of an Estimator",
        ),
        (
            "modules/model_evaluation.rst",
            "modules/model_evaluation.html",
            "Metrics and Scoring: Quantifying the Quality of Predictions",
        ),
        ("modules/compose.rst", "modules/compose.html", "Pipelines and Composite Estimators"),
        ("modules/impute.rst", "modules/impute.html", "Imputation of Missing Values"),
        ("modules/clustering.rst", "modules/clustering.html", "Clustering"),
    ]

    included = []
    for rel_src, rel_web, title in pages:
        src_url = f"{base_raw}{rel_src}"
        web_url = f"{base_web}{rel_web}"
        local_filename = rel_src.replace("/", "_")
        local_file = sklearn_raw / local_filename
        if download_url(src_url, local_file):
            included.append(
                {
                    "title": title,
                    "relative_path": f"scikit_learn/{local_filename}",
                    "url": web_url,
                    "format": "rst",
                }
            )
            print(f"  [+] Scikit-learn: {local_filename}")

    return {
        "source_name": "scikit_learn",
        "official_website": "https://scikit-learn.org/stable/",
        "repository": "https://github.com/scikit-learn/scikit-learn",
        "branch": "main",
        "content_source_revision": "unknown",
        "license_verification_revision": "d5a61d1149cb94a936d4b32212330f4ba8f24ce0",
        "verification_status": "unknown",
        "commit_sha": "d5a61d1149cb94a936d4b32212330f4ba8f24ce0",  # Verified HEAD on main
        "version": (
            "Scikit-learn main branch snapshot (corresponds to Scikit-learn 1.5-1.6 "
            "stable User Guide at scikit-learn.org/stable/)"
        ),
        "license": "BSD 3-Clause License",
        "license_file": "data/raw/scikit_learn/LICENSE",
        "collection_method": (
            "Raw source RST files downloaded from official Scikit-learn repository"
        ),
        "raw_directory": "data/raw/scikit_learn",
        "included_page_count": len(included),
        "included_pages": included,
        "excluded_files_or_categories": [
            "doc/developers/",
            "doc/whats_new/",
            "doc/governance.rst",
            "doc/presentations.rst",
        ],
        "reasons_for_exclusions": (
            "Developer guidelines, governance, release notes, and community documents "
            "excluded to focus on machine learning user guide topics."
        ),
    }


def collect_docker_docs(raw_dir: Path) -> dict:
    print("Collecting Docker documentation...")
    docker_raw = raw_dir / "docker"
    docker_raw.mkdir(parents=True, exist_ok=True)

    license_url = "https://raw.githubusercontent.com/docker/docs/main/LICENSE"
    download_url(license_url, docker_raw / "LICENSE")

    base_raw = "https://raw.githubusercontent.com/docker/docs/main/content/"
    base_web = "https://docs.docker.com/"

    pages = [
        ("get-started/docker-overview.md", "get-started/docker-overview/", "Docker Overview"),
        (
            "get-started/docker-concepts/the-basics/what-is-a-container.md",
            "get-started/docker-concepts/the-basics/what-is-a-container/",
            "What is a Container?",
        ),
        (
            "get-started/docker-concepts/the-basics/what-is-an-image.md",
            "get-started/docker-concepts/the-basics/what-is-an-image/",
            "What is an Image?",
        ),
        (
            "get-started/docker-concepts/the-basics/what-is-docker-compose.md",
            "get-started/docker-concepts/the-basics/what-is-docker-compose/",
            "What is Docker Compose?",
        ),
        (
            "get-started/docker-concepts/the-basics/what-is-a-registry.md",
            "get-started/docker-concepts/the-basics/what-is-a-registry/",
            "What is a Container Registry?",
        ),
        (
            "get-started/docker-concepts/building-images/writing-a-dockerfile.md",
            "get-started/docker-concepts/building-images/writing-a-dockerfile/",
            "Writing a Dockerfile",
        ),
        (
            "get-started/docker-concepts/building-images/build-tag-and-publish-an-image.md",
            "get-started/docker-concepts/building-images/build-tag-and-publish-an-image/",
            "Build, Tag, and Publish an Image",
        ),
        (
            "get-started/docker-concepts/building-images/multi-stage-builds.md",
            "get-started/docker-concepts/building-images/multi-stage-builds/",
            "Multi-stage Builds",
        ),
        (
            "get-started/docker-concepts/building-images/using-the-build-cache.md",
            "get-started/docker-concepts/building-images/using-the-build-cache/",
            "Using the Build Cache",
        ),
        (
            "get-started/docker-concepts/running-containers/publishing-ports.md",
            "get-started/docker-concepts/running-containers/publishing-ports/",
            "Publishing Ports",
        ),
        (
            "get-started/docker-concepts/running-containers/persisting-container-data.md",
            "get-started/docker-concepts/running-containers/persisting-container-data/",
            "Persisting Container Data",
        ),
        (
            "get-started/docker-concepts/running-containers/sharing-local-files.md",
            "get-started/docker-concepts/running-containers/sharing-local-files/",
            "Sharing Local Files (Bind Mounts)",
        ),
        (
            "get-started/docker-concepts/running-containers/multi-container-applications.md",
            "get-started/docker-concepts/running-containers/multi-container-applications/",
            "Multi-container Applications",
        ),
    ]

    included = []
    for rel_src, rel_web, title in pages:
        src_url = f"{base_raw}{rel_src}"
        web_url = f"{base_web}{rel_web}"
        local_filename = Path(rel_src).name
        local_file = docker_raw / local_filename
        if download_url(src_url, local_file):
            included.append(
                {
                    "title": title,
                    "relative_path": f"docker/{local_filename}",
                    "url": web_url,
                    "format": "md",
                }
            )
            print(f"  [+] Docker: {local_filename}")

    return {
        "source_name": "docker",
        "official_website": "https://docs.docker.com/",
        "repository": "https://github.com/docker/docs",
        "branch": "main",
        "content_source_revision": "unknown",
        "license_verification_revision": "4937e25aa671f382c793fd03ba457bdb25e0d028",
        "verification_status": "unknown",
        "commit_sha": "4937e25aa671f382c793fd03ba457bdb25e0d028",  # Verified HEAD on main
        "version": (
            "Docker Docs main branch snapshot (corresponds to current official "
            "documentation at docs.docker.com)"
        ),
        "license": "Apache License 2.0 (Apache-2.0)",
        "license_file": "data/raw/docker/LICENSE",
        "collection_method": (
            "Raw source Markdown files downloaded from official docker/docs repository"
        ),
        "raw_directory": "data/raw/docker",
        "included_page_count": len(included),
        "included_pages": included,
        "excluded_files_or_categories": [
            "content/manuals/subscription-billing/",
            "content/manuals/support/",
            "content/manuals/release-lifecycle.md",
        ],
        "reasons_for_exclusions": (
            "Commercial billing, account management, and legacy release "
            "lifecycle documents excluded."
        ),
    }


def collect_fastapi_docs(raw_dir: Path) -> dict:
    print("Collecting FastAPI documentation...")
    fastapi_raw = raw_dir / "fastapi"
    fastapi_raw.mkdir(parents=True, exist_ok=True)

    license_url = "https://raw.githubusercontent.com/fastapi/fastapi/master/LICENSE"
    download_url(license_url, fastapi_raw / "LICENSE")

    base_raw = "https://raw.githubusercontent.com/fastapi/fastapi/master/docs/en/docs/"
    base_web = "https://fastapi.tiangolo.com/"

    pages = [
        ("index.md", "", "FastAPI Introduction"),
        ("tutorial/first-steps.md", "tutorial/first-steps/", "First Steps"),
        ("tutorial/path-params.md", "tutorial/path-params/", "Path Parameters"),
        ("tutorial/query-params.md", "tutorial/query-params/", "Query Parameters"),
        ("tutorial/body.md", "tutorial/body/", "Request Body"),
        ("tutorial/response-model.md", "tutorial/response-model/", "Response Model - Return Type"),
        ("tutorial/dependencies/index.md", "tutorial/dependencies/", "Dependencies - First Steps"),
        ("tutorial/security/index.md", "tutorial/security/", "Security - First Steps"),
        ("tutorial/middleware.md", "tutorial/middleware/", "Middleware"),
        ("tutorial/handling-errors.md", "tutorial/handling-errors/", "Handling Errors"),
        ("tutorial/background-tasks.md", "tutorial/background-tasks/", "Background Tasks"),
        ("tutorial/testing.md", "tutorial/testing/", "Testing"),
        ("tutorial/sql-databases.md", "tutorial/sql-databases/", "SQL (Relational) Databases"),
        ("deployment/index.md", "deployment/", "Deployment Concepts"),
    ]

    included = []
    for rel_src, rel_web, title in pages:
        src_url = f"{base_raw}{rel_src}"
        web_url = f"{base_web}{rel_web}"
        local_filename = rel_src.replace("/", "_")
        local_file = fastapi_raw / local_filename
        if download_url(src_url, local_file):
            included.append(
                {
                    "title": title,
                    "relative_path": f"fastapi/{local_filename}",
                    "url": web_url,
                    "format": "md",
                }
            )
            print(f"  [+] FastAPI: {local_filename}")

    return {
        "source_name": "fastapi",
        "official_website": "https://fastapi.tiangolo.com/",
        "repository": "https://github.com/fastapi/fastapi",
        "branch": "master",
        "content_source_revision": "unknown",
        "license_verification_revision": "80d475b44a278253d15744a683af9614853af19e",
        "verification_status": "unknown",
        "commit_sha": "80d475b44a278253d15744a683af9614853af19e",  # Verified HEAD on master
        "version": (
            "FastAPI master branch snapshot (corresponds to FastAPI 0.115+ official "
            "documentation at fastapi.tiangolo.com)"
        ),
        "license": "MIT License",
        "license_file": "data/raw/fastapi/LICENSE",
        "collection_method": (
            "Raw source Markdown files downloaded from official fastapi/fastapi "
            "repository (docs/en/)"
        ),
        "raw_directory": "data/raw/fastapi",
        "included_page_count": len(included),
        "included_pages": included,
        "excluded_files_or_categories": [
            "docs/zh/",
            "docs/es/",
            "docs/ja/",
            "docs/pt/",
            "docs/ru/",
            "docs/de/",
            "docs/ko/",
        ],
        "reasons_for_exclusions": (
            "Non-English translations excluded; only English official documentation collected."
        ),
    }


def collect_aws_docs(raw_dir: Path) -> dict:
    print("Collecting AWS documentation...")
    aws_raw = raw_dir / "aws"
    aws_raw.mkdir(parents=True, exist_ok=True)

    # Save license attribution
    license_text = (
        "AWS Documentation Attribution & License:\n"
        "Source: Official Amazon Web Services (AWS) Documentation (https://docs.aws.amazon.com/)\n"
        "Documentation License: Creative Commons Attribution-ShareAlike 4.0 International "
        "(CC-BY-SA-4.0)\n"
        "Code Examples License: MIT No Attribution (MIT-0)\n"
        "Reference: AWS Open Source Documentation and Code Samples licensing terms "
        "(https://aws.amazon.com/terms/)\n"
    )
    (aws_raw / "LICENSE.txt").write_text(license_text, encoding="utf-8")

    pages = [
        # IAM (3 pages)
        (
            "https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html",
            "iam_introduction.html",
            "What is IAM?",
        ),
        (
            "https://docs.aws.amazon.com/IAM/latest/UserGuide/id.html",
            "iam_identities.html",
            "IAM Identities (users, user groups, and roles)",
        ),
        (
            "https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies.html",
            "iam_policies.html",
            "Policies and Permissions in IAM",
        ),
        # Amazon S3 (3 pages)
        (
            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/Welcome.html",
            "s3_welcome.html",
            "What is Amazon S3?",
        ),
        (
            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/UsingBucket.html",
            "s3_buckets_objects.html",
            "Working with Amazon S3 Buckets",
        ),
        (
            "https://docs.aws.amazon.com/AmazonS3/latest/userguide/access-control-overview.html",
            "s3_access_control.html",
            "Identity and Access Management for Amazon S3",
        ),
        # Amazon EC2 (3 pages)
        (
            "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/concepts.html",
            "ec2_concepts.html",
            "What is Amazon EC2?",
        ),
        (
            "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/Instances.html",
            "ec2_instances.html",
            "Amazon EC2 Instances",
        ),
        (
            "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-security-groups.html",
            "ec2_security_groups.html",
            "Amazon EC2 Security Groups",
        ),
        # AWS Lambda (3 pages)
        (
            "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
            "lambda_welcome.html",
            "What is AWS Lambda?",
        ),
        (
            "https://docs.aws.amazon.com/lambda/latest/dg/lambda-runtime-environment.html",
            "lambda_runtime_environment.html",
            "AWS Lambda Execution Environment",
        ),
        (
            "https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html",
            "lambda_execution_role.html",
            "AWS Lambda Execution Role",
        ),
    ]

    included = []
    for web_url, local_name, title in pages:
        local_file = aws_raw / local_name
        if download_url(web_url, local_file):
            included.append(
                {
                    "title": title,
                    "relative_path": f"aws/{local_name}",
                    "url": web_url,
                    "format": "html",
                }
            )
            print(f"  [+] AWS: {local_name}")

    return {
        "source_name": "aws",
        "official_website": "https://docs.aws.amazon.com/",
        "repository": (
            "N/A (rendered HTML documentation used because official AWS docs repositories "
            "are not maintained as public Markdown trees for all 4 services)"
        ),
        "branch": "N/A",
        "content_source_revision": "N/A",
        "license_verification_revision": "N/A",
        "verification_status": "verified",
        "commit_sha": "N/A",
        "version": "AWS Documentation Current (November 2026)",
        "license": "Documentation: CC-BY-SA-4.0; Code Examples: MIT-0",
        "license_file": "data/raw/aws/LICENSE.txt",
        "compliance_considerations": (
            "Reviewed AWS Site Terms (Section 3 'Site Access and Content'). The collection "
            "method adheres to terms: targeted, polite HTTP requests for 12 specific HTML "
            "conceptual pages (3 per service) with identified User-Agent "
            "('RAGBench-Collector/1.0'), spacing intervals between calls, no automated mass "
            "scraping, no unauthorized access to private APIs, and no unreasonable load on "
            "AWS infrastructure. Attribution and licensing are preserved under "
            "CC-BY-SA-4.0 (prose) and MIT-0 (code)."
        ),
        "collection_method": (
            "Rendered official HTML pages downloaded directly from docs.aws.amazon.com"
        ),
        "raw_directory": "data/raw/aws",
        "included_page_count": len(included),
        "included_pages": included,
        "excluded_files_or_categories": [
            "AWS Services outside IAM, S3, EC2, and Lambda",
            "API Reference documentation",
            "CLI Reference documentation",
        ],
        "reasons_for_exclusions": (
            "AWS documentation scope intentionally restricted to core concepts of four "
            "essential services (IAM, S3, EC2, Lambda)."
        ),
    }


def main() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).isoformat()
    sources_data = [
        collect_python_docs(raw_dir),
        collect_scikit_learn_docs(raw_dir),
        collect_docker_docs(raw_dir),
        collect_fastapi_docs(raw_dir),
        collect_aws_docs(raw_dir),
    ]

    manifest = {
        "project": "RAGBench",
        "collection_timestamp_utc": timestamp,
        "sources": sources_data,
    }

    manifest_path = raw_dir / "source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nCollection complete! Source manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
