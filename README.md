# sdt-pvdb-app
A [Marimo app](https://marimo.io/?site) for applying Solar Data Tools to GISMo's photovolaic database (PVDB)

## Installation

Clone the repository and then run `pip install -r requirements.txt`, preferably in a fresh (minimal existing packages) Python 3.10+ virtual environment.

## Data API license

This application connects to an open-source data API that requires registration. Please visit https://pvdb.slacgismo.org, click "sign in" and register for a new account. In addition, please send an email to slacgismotutorials@gmail.com from your registration email, indicating your wish to test out this application. Thank you!

After being approved, you will receive a login to https://pvdb.slacgismo.org to retreive your API key. The final step is to set this key as an environment variable on your system, by adding it to your `.bashrc` or `.zshrc` profile as follows:

```
## Setting personal API key for SunPower Redshift DB
export REDSHIFT_API_KEY=[your key here]
```

After your personal API key has been set an an environment variable, you will be able to use the app to pull data for analysis from PVDB.

## Running Marimo app

After installation and setting up your API key, simply execute the following command in the project folder:

```
marimo run app.py
```

## Using the app

You can press the "execute" button right away to load in an example data set. It should take about 2-4 minutes to fully load, process, and visualize the data set on standard laptops. From there you can explore the tabs to see other views of this data set, or load in new data, by selecting a site from the table and a system ID from the dropdown menu in the upper left. Data and results are cached during a session for quick reloaded (<5 seconds). Your cache list is visible in the secondary tab, under the site table. Use the toggle switch in the upper left to choose between loading data from the full site list or the cache list.
