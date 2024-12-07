# AlgoTrade Project

This repository contains an algorithmic trading system built using the Freqtrade framework. The project is containerized using Docker for easy deployment and consistent environments across different systems.

## Prerequisites

- Docker
- Docker Compose
- Git

## Getting Started

1. Clone this repository:
```bash
git clone https://github.com/yourusername/algoTrade.git
cd algoTrade
```

2. Copy and configure your config file:
```bash
cp config.json.example config.json
```
Edit `config.json` with your preferred settings and exchange API credentials.

## Using Docker Compose

This project uses Docker Compose for managing the Freqtrade environment. Instead of using the `freqtrade` command directly, we use `docker-compose run --rm` for all operations. Here's why:

- **Consistent Environment**: Ensures all dependencies and versions are exactly the same across different systems
- **Isolation**: Keeps your system clean by running everything in containers
- **No Installation Required**: No need to install Python or any dependencies on your host system
- **Easy Cleanup**: The `--rm` flag automatically removes the container when the command finishes

### Common Commands

Here are the most common commands you'll use, translated from Freqtrade to Docker Compose:

#### Download Price Data
```bash
docker-compose run --rm freqtrade download-data --timeframe 1m --timerange 20180101-20230615
```

#### Backtesting
```bash
docker-compose run --rm freqtrade backtesting --strategy YourStrategy
```

#### Run Trading Bot
```bash
docker-compose run --rm freqtrade trade --strategy YourStrategy
```

#### Create New Strategy
```bash
docker-compose run --rm freqtrade new-strategy --strategy AwesomeStrategy
```

#### List Strategies
```bash
docker-compose run --rm freqtrade list-strategies
```

#### Show Plots
```bash
docker-compose run --rm freqtrade plot-dataframe --strategy YourStrategy
```

## Project Structure

```
algoTrade/
├── config.json           # Main configuration file
├── docker-compose.yml    # Docker compose configuration
├── strategies/          # Trading strategies
├── user_data/          # User specific data
│   ├── data/          # Downloaded historical data
│   └── models/        # Trained models (if using ML)
└── notebooks/         # Jupyter notebooks for analysis
```

## Development

To develop and test new strategies:

1. Create a new strategy:
```bash
docker-compose run --rm freqtrade new-strategy --strategy MyNewStrategy
```

2. Edit the strategy file in the `user_data/strategies` directory

3. Backtest your strategy:
```bash
docker-compose run --rm freqtrade backtesting --strategy MyNewStrategy
```

## Monitoring

To monitor your running bot:

1. Access the Freqtrade REST API (if enabled in config):
```bash
docker-compose run --rm freqtrade api
```

2. View the logs:
```bash
docker-compose logs -f
```

## Troubleshooting

If you encounter issues:

1. Check the logs:
```bash
docker-compose logs -f
```

2. Ensure your config.json is properly configured

3. Verify your exchange API credentials

4. Make sure you have sufficient funds in your exchange account

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
