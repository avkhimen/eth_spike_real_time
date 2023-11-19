# ETH Spike Real Time

ETH spike real time monitors the Ethereum price and sends a text message notification when ETH is about to spike against the Bitcoin currency.

The model used to determine the Ethereum spike is based on KNN regression. The `model_training.py` file contains the details of the model training. 90 4-hour input steps together 2ith 17 nearest-neighbor points, together with undersampling to balance the dataset, produced model accuracies in the 75% range. Oversampling resulted in similar accuracies.

## Installation

The application is fully containerized.

## Usage

```shell
docker compose up --build
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)