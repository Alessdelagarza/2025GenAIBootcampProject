# Project Description
This project showcases the application of GenAI concepts learned during the Slalom GenAI bootcamp. It interprets user prompts to understand the context and uses that context to select from a predefined list of options, applying various effects to a video feed.


# Set Up Instructions
## Package Management with Pipenv

We use Pipenv to manage our project dependencies. Pipenv simplifies the process of managing Python packages, virtual environments, and dependency resolution.

### Steps to Set Up Pipenv

1. **Install Pipenv**:
    ```sh
    pip install pipenv
    ```

2. **Install Project Dependencies**:
    Navigate to the project root directory and run:
    ```sh
    pipenv install
    ```

3. **Set Up a Virtual Environment**:
    To create and activate a virtual environment, use:
    ```sh
    pipenv shell
    ```

These steps ensure that all required packages are installed and that the project runs in an isolated environment.

### Testing Configurations

To ensure your configuration is set up correctly, run the manual tests:
```sh
pipenv run test-configs
```

### Try the Prototype

Once you've successfully passed the configuration tests, you're ready to experience the exciting initial prototype of our application! Give it a try with this command:
```sh
pipenv run try-prototype
```


