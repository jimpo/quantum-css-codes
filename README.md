# Quantum CSS Codes

This is a software implementation of fault tolerant quantum computation (FTQC) using Calderbank-Steane-Shor (CSS) codes. [CSS codes](https://en.wikipedia.org/wiki/CSS_code) are a family of quantum error correcting codes that are constructed from pairs of classical error correcting codes. They are a special class of stabiliser codes where the stabiliser group can be generated from Pauli terms that are a product of only X or only Z terms.

This project was written for [Stanford 269Q](https://cs269q.stanford.edu/). The library is written in Python using Rigetti's [Forest SDK](http://docs.rigetti.com/en/stable/). The library includes an interface to generate a CSS code with its stabilisers, Pauli operators, encoding network, etc. given two appropriate binary linear codes. Using the CSS code library, the project also includes a function to rewrite a pyQuil `Program` into one that implements the same logic in a fault tolerant manner by encoding each qubit with a CSS code.

See module docstrings for documentation.

## Installation

You will need:

- [Pipenv](https://docs.pipenv.org/en/latest/)
- [Forest SDK](http://docs.rigetti.com/en/stable/start.html)

Install dependencies with `$ pipenv install`.

## Running tests

Run tests with

```bash
$ pipenv run python -m unittest
```
