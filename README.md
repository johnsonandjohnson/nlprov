<p align="center">
  <img width="250" src="https://github.com/johnsonandjohnson/nlprov/raw/master/images/nlplogo.png">
</p>

# __NLProv__: Natural Language Processing Tool

<p align="left">
 <a href="https://github.com/johnsonandjohnson/nlprov/actions">
   <img src="https://github.com/johnsonandjohnson/nlprov/workflows/Build,%20Test,%20and%20Package/badge.svg" />
 </a>
 <a href="https://codecov.io/gh/johnsonandjohnson/nlprov">
   <img src="https://codecov.io/gh/johnsonandjohnson/nlprov/branch/master/graph/badge.svg" />
 </a>
</p>

NLProv is a Python library developed by Johnson & Johnson's Advanced Analytics 
team that combines existing libraries for common Natural Language Processing tasks.
It combines several existing open-source libraries such as pandas, spaCy, and
scikit-learn to make a pipeline that is ready to process text data. There are
many user defined parameters depending on your type of project such as the
ability to choose stemming or lemmatization. Or, you might want to define
explicitly what to substitute with NaN text fields. Overall, it is a way to get
you started in your NLP task, no matter what you need.

A tutorial on how to use this package can be found [here](tutorial.ipynb).

## Installation Instructions

### To install this package locally:
   - Clone the repository from the master branch
   - Within the terminal, navigate into the repo directory
   - Run the following command to install nlp_python_package locally:
        ```shell
        pip install .
        python -m spacy download en_core_web_sm 
        # en_core_web_sm is required for spacy text pre-processing
        # The . can be replaced with a path to the directory instead
        ```
   - For more information on installing packages using pip, click [here](https://pip.pypa.io/en/stable/reference/pip_install/).

### Contributing 
- To help develop this package, you'll need to install a conda virtual 
environment defined by our dev_environment.yml file using the below command.

  ```shell
  conda env create -f dev_environment.yml
  ```
  - Then, just activate the environment when attempting to develop or run tests 
  using the below command.

    ```shell
    conda activate nlp_env
    ```

  - When you're all done developing or testing, just deactivate the environment 
  with the below command.

    ```shell
    conda deactivate
    ```

## Docker Configuration
- This codebase is dockerized to build, run all of the unit tests using `pytest`, and perform pip packaging.
  - In order to run the docker container, ensure you have [Docker](https://www.docker.com/products/docker-desktop) 
  installed and running on your local machine.
  - To start the docker container locally, simply navigate to the root of the 
  project directory and type:
  ```shell
  docker-compose up --build
  ```
  - Note: `docker-compose` is included in the Docker desktop installation link 
  above for MacOS and Windows based systems. If you have issues executing 
  `docker-compose`, [Navigate Here](https://docs.docker.com/compose/install/) 
  to ensure docker-compose is supported on your system.
  - A Notey-er note: You can use `docker-compose up --build` during development 
  to quickly run the tests after code changes without setting up/running a local 
  conda environment.

## GitHub Action CI Configuration
- Every commit to this repository will trigger a build in GitHub Actions following the
 .github/workflows/pythonapp.yml located in the root of this project.
  - GitHub Actions is used to build and lint the NLProv package, run the tests, and perform pip packaging.
  - If the environment name or version changes, the pythonapp.yml file will need to be updated to 
  follow the new pattern.
  
## Our Workflow
- Our Methods and Tools
  - Style Guide - [PEP8 / pycodestyle](https://www.python.org/dev/peps/pep-0008/)
  - Git Strategy - [Git Flow](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow)

## Upcoming Features
Here is a roadmap of features to be implemented in this package. If you have any
ideas for additional features, please let us know!

* Preprocessing
    * Ability to use custom stop words
    * Incorporation of bi-grams
    * Ability for user to chose which langauge detection package to use
* Vectorization
    * spaCy pre-trained models
    * spaCy custom models
* Similarity Metrics
    * Additional pairwise distances
    * Levenshtein Distance
    * Word Mover's Distance
* Visualizations
    * TF-IDF
    * Jaccard

