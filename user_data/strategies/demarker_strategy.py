from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta
import pandas as pd
import numpy as np
from datetime import datetime, time
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_minutes
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class DeMarkerCCIStrategy(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '1h'
    stoploss = -0.08
    can_short = True
    use_exit_signal = True

    # Lock-Settings hinzufügen/ändern
    lock_pair_after_exit = False      # Kein Auto-Lock nach Trade
    exit_trade_timeout = 0            # Kein Timeout nach Exit
    ignore_roi_if_entry_signal = False
    ignore_buying_expired_candle_after = 0
    
    # PairLock Settings
    use_custom_pairlock = False
    pairlock_reset_after = 0
    pairlock_expire_after = 0

    # Trailing Stop Konfiguration
    trailing_stop = True
    trailing_stop_positive = 0.01  # Aktiviere Trailing Stop bei 1% Profit
    trailing_stop_positive_offset = 0.02  # Offset von 2%
    trailing_only_offset_is_reached = True  # Warte auf Offset


  # Adjusted stake settings
    stake_amount = 250.0  # Set base stake amount to match minimum requirement

    minimal_roi = {
        "0": 0.04,
     #   "10": 0.03,
     #   "20": 0.02,
     #   "30": 0.01
    }

  # Exchange Minimums und Position Sizes
    position_configs = {
        'BNB/USDT:USDT': {
            'min_stake': 150.0,
            'min_amount': 0.2,
            'amount_precision': 2,
            'price_precision': 2
        },
        'SOL/USDT:USDT': {
            'min_stake': 150.0,
            'min_amount': 0.5,
            'amount_precision': 2,
            'price_precision': 2
        },
        'ETH/USDT:USDT': {
            'min_stake': 150.0,
            'min_amount': 0.1,
            'amount_precision': 3,
            'price_precision': 2
        }
    }

    unfilledtimeout = {
        "entry": 10,  # 10 Minuten statt 1 Stunde
        "exit": 10,
        "unit": "minutes"
    }

    process_only_new_candles = True
    startup_candle_count = 50

    hedge_config = {
                'enabled': True,
                'hedge_ratio': 1.0,          # Volle Gegenposition
                'activation_loss': -0.025,   # Stop bei -2.5%
                'profit_close': 0.01,       # Original Trade schließen bei +0.5%
                'hedge_stoploss': -0.03      # Hedge-Position Stoploss bei -1%
            }

    _last_candle_seen_per_pair = {}
    _last_cleanup_time = 0

    custom_info = {
        'magic_number': 11111,
        'qqe_rsi_period': 12,
        'qqe_sf': 4,
        'dem_period': 8,
        'cci_period': 10,
        'period': 14,
        'profit_target_coef': 2.0,
        'stoploss_coef': 1.2,
        'long_order_timeout': 30,
        'short_order_timeout': 30,
        'profit_target1': 3.0,
        'stoploss1': 1.5,
        'trailing_stop1': 120,
        'last_trade_timestamps': {},
        'entry_timeout': 2,
        'price_adjust_factor': 0.002,
        'cooldown_period': 0.25,  # Cooldown in Stunden
        'last_trade_exits': {}  # Speichert die letzten Trade-Exit-Zeitpunkte
    }

    order_types = {
        'entry': 'limit',
        'exit': 'limit', 
        'emergency_exit': 'market',
        'force_exit': 'market',
        'force_entry': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 60,
        'stoploss_on_exchange_limit_ratio': 0.99
    }


    order_time_in_force = {
        'entry': 'GTC',
        'exit': 'GTC'
    }

    # Minimum trade size settings for Binance Futures
    minimum_trade_size = {
        'BTC/USDT:USDT': 0.001,
        'ETH/USDT:USDT': 0.01,
        'default': 0.01
    }
    # Adjusted stake settings
    stake_amount = 200.0  # Fester Betrag pro Trade
    # Pair-specific minimum stakes
  # Angepasste Mindest-Stakes für jedes Pair
    pair_min_stakes = {
        'BTC/USDT:USDT': 200.0,
        'ETH/USDT:USDT': 200.0,
        'SOL/USDT:USDT': 200.0,  # Angepasst an Exchange-Minimum
        'BNB/USDT:USDT': 200.0,
        'default': 200.0
    }

    @property
    def max_entry_position_adjustment(self):
        return -1  # Deaktiviert Position-Adjustments
        
    def custom_pair_lock(self, pair: str, timeframe: str, candle_date: datetime, **kwargs) -> Optional[datetime]:
        return None  # Erlaubt immer Trading


    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: float, max_stake: float,
                            **kwargs) -> Optional[float]:
        """
        Adjust position size and handle position management
        """
        try:
            # Check if this is a hedge position
            if trade.tags is not None and isinstance(trade.tags, dict) and any('hedge_stop' in str(tag) for tag in trade.tags):
                return None  # Don't adjust hedge positions
                
            # Handle regular position adjustments
            if current_profit <= self.hedge_config['activation_loss']:
                # Calculate hedge position size
                hedge_stake = trade.stake_amount * self.hedge_config['hedge_ratio']
                hedge_stake = min(max(hedge_stake, min_stake), max_stake)
                
                # Create hedge tag
                hedge_tag = f"hedge_stop_{trade.id}_activation_{abs(self.hedge_config['activation_loss'])}"
                
                # Update trade tags
                if trade.tags is None:
                    trade.tags = {}
                trade.tags['hedge'] = hedge_tag
                
                logger.info(f"Opening hedge position: {hedge_tag} with stake: {hedge_stake}")
                return hedge_stake
                
            return None

        except Exception as e:
            logger.error(f"Error in adjust_trade_position: {e}")
            return None



    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        if len(dataframe) == 0:
            return dataframe

        try:
            original_close = dataframe['close'].copy()

            # Optimierte Datentypen für bessere Performance
            for col in dataframe.select_dtypes(include=['float64']).columns:
                if col != 'close':
                    dataframe[col] = dataframe[col].astype('float32')

          
        # Berechnung des weighted_close
            dataframe['weighted_close'] = (dataframe['high'] + dataframe['low'] + dataframe['close'] * 2) / 4
            

            # RSI und QQE
            rsi_period = self.custom_info['qqe_rsi_period']
            sf = self.custom_info['qqe_sf']
            wilders = 4.236

            dataframe['rsi'] = ta.RSI(dataframe['weighted_close'], timeperiod=rsi_period)
            dataframe['qqe_ema'] = ta.EMA(dataframe['rsi'], timeperiod=sf)
            dataframe['qqe'] = ta.EMA(dataframe['qqe_ema'], timeperiod=int(wilders))

            # Verbesserte Trend-Erkennung
            dataframe['qqe_rising'] = (
                (dataframe['qqe'] > dataframe['qqe'].shift(3)) 
            )

            # Optimierte DeMarker
            high_dm = np.where(
                (dataframe['high'] > dataframe['high'].shift(1)),
                dataframe['high'] - dataframe['high'].shift(1),
                0
            )
            low_dm = np.where(
                (dataframe['low'] < dataframe['low'].shift(1)),
                dataframe['low'].shift(1) - dataframe['low'],
                0
            )

            dataframe['demarker'] = pd.Series(high_dm).rolling(
                window=self.custom_info['dem_period']
            ).sum() / (
                pd.Series(high_dm).rolling(window=self.custom_info['dem_period']).sum() +
                pd.Series(low_dm).rolling(window=self.custom_info['dem_period']).sum()
            )

            # Verbesserte DeMarker Signale
            dataframe['demarker_direction_change'] = (
                (dataframe['demarker'].shift(3) > dataframe['demarker'].shift(4)) &  # War steigend
                (dataframe['demarker'].shift(2) < dataframe['demarker'].shift(3))    # Ist jetzt fallend
            )

            # Optimierte CCI mit gewichtetem Preis
            dataframe['cci'] = ta.CCI(
                dataframe['high'],
                dataframe['low'],
                dataframe['weighted_close'],
                timeperiod=self.custom_info['cci_period']
            )

            # CCI direction change downwards
            dataframe['cci_direction_change'] = (
                (dataframe['cci'].shift(1) > dataframe['cci'].shift(2)) &  # War steigend
                (dataframe['cci'] < dataframe['cci'].shift(1))             # Ist jetzt fallend
            )

            # EMAs für Trendbestätigung
            dataframe['ema_20'] = ta.EMA(dataframe['weighted_close'], timeperiod=20)
            dataframe['ema_50'] = ta.EMA(dataframe['weighted_close'], timeperiod=50)

            # ATR und Volatilitäts-Berechnung
            dataframe['atr'] = ta.ATR(dataframe['high'], dataframe['low'], dataframe['close'],
                                    timeperiod=14)

            # Dynamischer Volatilitäts-Multiplikator
            dataframe['volatility'] = dataframe['atr'] / dataframe['close']
            dataframe['vol_ma'] = dataframe['volatility'].rolling(window=50).mean()

            # Angepasste Bänder
            hl2 = (dataframe['high'] + dataframe['low']) / 2
            volatility_mult = np.where(
                dataframe['volatility'] > dataframe['vol_ma'],
                1.6,  # Erhöhter Multiplikator bei hoher Volatilität
                1.2   # Normaler Multiplikator
            )

            dataframe['upperband'] = hl2 + (volatility_mult * dataframe['atr'])
            dataframe['lowerband'] = hl2 - (volatility_mult * dataframe['atr'])
            dataframe['dynamic_stoploss'] = dataframe['lowerband']

            

            # MACD für zusätzliche Trendbestätigung
            dataframe['macd'], dataframe['macdsignal'], _ = ta.MACD(
                dataframe['weighted_close'],
                fastperiod=12,
                slowperiod=26,
                signalperiod=9
            )

            # ADX für Trendstärke
            dataframe['adx'] = ta.ADX(
                dataframe['high'],
                dataframe['low'],
                dataframe['close'],
                timeperiod=14
            )

            # MFI für Volumen-Preisbestätigung
            dataframe['mfi'] = ta.MFI(
                dataframe['high'],
                dataframe['low'],
                dataframe['close'],
                dataframe['volume'],
                timeperiod=14
            )

            # Zusätzliche Momentum-Indikatoren
            dataframe['roc'] = ta.ROC(dataframe['close'], timeperiod=10)
            dataframe['mom'] = ta.MOM(dataframe['close'], timeperiod=10)

            dataframe['close'] = original_close

            return dataframe

        except Exception as e:
            logger.error(f"Critical error in populate_indicators: {str(e)}")
            return dataframe




    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            if len(dataframe) == 0:
                return dataframe

            dataframe.loc[:, 'enter_long'] = 0
          #  dataframe.loc[:, 'enter_short'] = 0

            try:
                # Prüfe Cooldown für das aktuelle Pair
                current_time = datetime.fromtimestamp(dataframe.iloc[-1]['date'].timestamp())
                if self.check_cooldown(metadata['pair'], current_time):
                    return dataframe
                last_candle = dataframe.iloc[-1]  # Hier wird last_candle definiert
                logger.info(f"Pair: {metadata['pair']} - Analyzing candle: QQE: {last_candle['qqe']:.2f}, "
                        f"ADX: {last_candle['adx']:.2f}, DeMarker: {last_candle['demarker']:.2f}")


                # Simplified entry conditions
                long_conditions = (
                    dataframe['qqe_rising'] &
                    (dataframe['adx'] > 20) &
                    (dataframe['mfi'] < 70)  # Verhindere Einstieg bei Überkauf
                )
                 # Log der Long-Bedingungen
                logger.info(f"Pair: {metadata['pair']} - Long conditions: "
                        f"QQE Rising: {last_candle['qqe_rising']}, "
                        f"ADX > 12: {last_candle['adx'] > 12}")

               # short_conditions = (
                #    dataframe['demarker_direction_change'] &
                 #    ~long_conditions &
                   #  dataframe['ma_cross_short']
                   # dataframe['cci_direction_change'] #&
                  #  (dataframe['close'] < dataframe['ema_20']) #&
                  # (dataframe['ema_20'] < dataframe['ema_50']) #&
                  #  (dataframe['adx'] < 30) #&
                   # (dataframe['macd'] < dataframe['macdsignal']) &
                   # (dataframe['mfi'] > 30) &
                   # (dataframe['roc'] < 0) &
                   # (dataframe['mom'] < 0) &
                   # (dataframe['volume'] > dataframe['volume_ma'] * 1.2)
             #   )

                            # Log der Short-Bedingungen
                logger.info(f"Pair: {metadata['pair']} - Short conditions: "
                        f"DeMarker Change: {last_candle['demarker_direction_change']}, "
                        f"ADX < 30: {last_candle['adx'] < 30}")


                dataframe.loc[long_conditions, 'enter_long'] = 1
                #dataframe.loc[short_conditions, 'enter_short'] = 1
                 # Log der finalen Entscheidung
                logger.info(f"Pair: {metadata['pair']} - Trade signals: "
                        f"Long: {bool(dataframe.iloc[-1]['enter_long'])}")
                        #f"Short: {bool(dataframe.iloc[-1]['enter_short'])}")


            except Exception as e:
                logger.error(f"Error in populate_entry_trend: {str(e)}")

            return dataframe



    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                proposed_leverage: float, max_leverage: float, entry_tag: str, side: str, **kwargs) -> float:
        """
        Customize leverage for each new trade. This runs before a new entry order is placed.
        
        :param pair: Pair that's currently analyzed
        :param current_time: datetime object, containing the current datetime
        :param current_rate: Rate, calculated based on pricing settings in exit_pricing.
        :param proposed_leverage: A leverage proposed by the bot.
        :param max_leverage: Max leverage allowed on the exchange.
        :param entry_tag: Optional entry_tag if provided with the buy signal.
        :param side: 'long' or 'short' - indicating the direction of the proposed trade
        """
        try:
            # Check if this is a hedge trade
            is_hedge = entry_tag is not None and 'hedge_stop' in entry_tag
            
            # For hedge positions, maintain the same leverage as the original position
            if is_hedge:
                try:
                    # Extract original trade ID from hedge tag
                    original_trade_id = int(entry_tag.split('_')[2])
                    original_trades = Trade.get_trades([Trade.id == original_trade_id]).all()
                    
                    if original_trades and len(original_trades) > 0:
                        original_trade = original_trades[0]
                        return original_trade.leverage
                except (ValueError, IndexError, AttributeError) as e:
                    logger.warning(f"Could not get original trade leverage for hedge: {e}")
                    return 1.0
            
            # For non-hedge trades, use default leverage
            return 1.0

        except Exception as e:
            logger.error(f"Error in leverage function: {e}")
            return 1.0

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                current_rate: float, current_profit: float, **kwargs) -> bool:
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0 or not current_rate:
                return False

            # Profit-Taking für alle Trades
            if current_profit >= 0.04:  # 4% Profit
                logger.info(f"Closing trade at profit target: {current_profit}")
                return True

            # Hedge-Position Management
            if trade.is_open and hasattr(trade, 'trading_mode') and trade.trading_mode == 'spot':
                trade_tags = trade.tags if hasattr(trade, 'tags') else {}
                
                if trade_tags and 'hedge_stop' in str(trade_tags):
                    try:
                        original_trade_id = int(str(trade_tags).split('_')[2])
                        original_trades = Trade.get_trades([Trade.id == original_trade_id]).all()
                        
                        if original_trades and len(original_trades) > 0:
                            original_trade = original_trades[0]
                            original_profit = original_trade.calc_profit_ratio(current_rate)
                            
                            # Dynamische Hedge-Schließung basierend auf Original-Trade-Profit
                            if original_profit > 0.02:  # 2% Profit
                                logger.info(f"Closing hedge - original trade in good profit: {original_profit}")
                                return True
                            
                            # Progressive Stoploss für Hedge
                            hedge_stoploss = self.hedge_config['hedge_stoploss']
                            if current_profit < -0.01:  # -1%
                                # Verschärfter Stoploss bei kleineren Verlusten
                                hedge_stoploss = min(hedge_stoploss * 1.5, -0.04)  # Max -4%
                            
                            if current_profit <= hedge_stoploss:
                                logger.info(f"Closing hedge on dynamic stoploss: {current_profit}")
                                return True
                    except (ValueError, IndexError) as e:
                        logger.error(f"Error processing hedge trade tags: {e}")
                        return False

                # Original Trade Management
                else:
                    # Suche nach aktiven Hedge-Positionen
                    hedge_trades = Trade.get_trades([
                        Trade.is_open.is_(True),
                        Trade.tags.contains(f'hedge_stop_{trade.id}')
                    ]).all()
                    
                    if hedge_trades and len(hedge_trades) > 0:
                        # Profit-Taking für Original-Position
                        if current_profit > 0.03:  # 3% Profit
                            logger.info(f"Closing profitable original trade with active hedge: {current_profit}")
                            return True
                    else:
                        # Wenn kein Hedge aktiv ist, prüfe auf Hedge-Aktivierung
                        if current_profit <= self.hedge_config['activation_loss']:
                            # Lasse Position offen, damit Hedge eröffnet werden kann
                            return False

            return False

        except Exception as e:
            logger.error(f"Error in custom_exit: {e}")
            return False

    def custom_entry_price(self, pair: str, current_time: datetime,
                        proposed_rate: float, entry_tag: Optional[str],
                        side: str, **kwargs) -> float:
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) == 0:
                return proposed_rate
                
            last_candle = dataframe.iloc[-1].squeeze()
            
            # Berechne einen leicht angepassten Preis basierend auf der Richtung
            adjustment = 0.001  # 0.1% Preisanpassung
            
            if side == "long":
                entry_price = last_candle['close'] * (1 + adjustment)
            else:
                entry_price = last_candle['close'] * (1 - adjustment)
            
            logger.info(f"""
            Entry Price Calculation für {pair}:
            - Side: {side}
            - Current Close: {last_candle['close']:.8f}
            - Adjusted Price: {entry_price:.8f}
            """)
            
            return entry_price

        except Exception as e:
            logger.error(f"Error in custom_entry_price: {e}")
            return proposed_rate


    def check_cooldown(self, pair: str, current_time: datetime) -> bool:
        try:
            if pair not in self.custom_info['last_trade_exits']:
                return False
            last_exit_time = self.custom_info['last_trade_exits'][pair]
            cooldown_end_time = last_exit_time + pd.Timedelta(hours=0.25)  # auf 15 Minuten reduzieren
            logger.info(f"Checking cooldown for {pair}: {current_time} < {cooldown_end_time}")
            return current_time < cooldown_end_time
        except Exception as e:
            logger.error(f"Error in check_cooldown: {str(e)}")
            return False


    def cleanup_data(self, pair: str, current_time) -> None:
            try:
                current_timestamp = pd.Timestamp(current_time).timestamp()
                cleanup_interval = 3600

                if current_timestamp - self._last_cleanup_time > cleanup_interval:
                    current_pairs = self.dp.current_whitelist()

                    # Cleanup für last_trade_timestamps
                    if 'last_trade_timestamps' in self.custom_info:
                        self.custom_info['last_trade_timestamps'] = {
                            k: v for k, v in self.custom_info['last_trade_timestamps'].items()
                            if k in current_pairs
                        }

                    # Cleanup für last_trade_exits
                    if 'last_trade_exits' in self.custom_info:
                        # Entferne alte Cooldown-Einträge
                        cooldown_threshold = current_time - pd.Timedelta(hours=self.custom_info['cooldown_period'])
                        self.custom_info['last_trade_exits'] = {
                            k: v for k, v in self.custom_info['last_trade_exits'].items()
                            if v > cooldown_threshold and k in current_pairs
                        }

                    self._last_candle_seen_per_pair = {
                        k: v for k, v in self._last_candle_seen_per_pair.items()
                        if k in current_pairs
                    }

                    self._last_cleanup_time = current_timestamp

            except Exception as e:
                logger.error(f"Error in cleanup_data: {str(e)}")



    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            if len(dataframe) == 0:
                return dataframe

            dataframe.loc[:, 'exit_long'] = 0

            try:
                # Simplified exit conditions - mainly rely on trailing stop
                exit_conditions = (
                    (dataframe['adx'] < 10) &  # Trend wird sehr schwach
                    (dataframe['close'] < dataframe['ema_50'])  # Unter langfristigem Trend
                )

                dataframe.loc[exit_conditions, 'exit_long'] = 1

            except Exception as e:
                logger.error(f"Error in populate_exit_trend: {str(e)}")

            return dataframe

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                        proposed_stake: float, min_stake: Optional[float], max_stake: float,
                        leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        try:
            if not current_rate:
                return proposed_stake

            # Get pair config or use defaults
            pair_config = self.position_configs.get(pair, {
                'min_stake': 150.0,
                'min_amount': 0.1,
                'amount_precision': 3,
                'price_precision': 2
            })

            # Calculate minimum stake based on minimum amount
            min_stake_from_amount = pair_config['min_amount'] * current_rate
            
            # Use the larger of minimum stakes
            effective_min_stake = max(pair_config['min_stake'], min_stake_from_amount)
            
            # Calculate initial position size
            position_size = max(
                pair_config['min_amount'],
                proposed_stake / current_rate
            )
            
            # Round position size to pair precision
            position_size = round(position_size, pair_config['amount_precision'])
            
            # Calculate final stake amount
            final_stake = position_size * current_rate
            
            logger.info(f"""
            Detailed stake calculation for {pair}:
            - Current rate: {current_rate}
            - Initial position size: {position_size}
            - Min required stake: {effective_min_stake}
            - Final stake amount: {final_stake}
            - Position precision: {pair_config['amount_precision']}
            """)
            
            # Validate final stake
            if final_stake < effective_min_stake:
                logger.info(f"Stake too small for {pair}: {final_stake} < {effective_min_stake}")
                return 0
                
            if final_stake > max_stake:
                logger.info(f"Stake too large for {pair}: {final_stake} > {max_stake}")
                return 0
                
            return final_stake

        except Exception as e:
            logger.error(f"Error in custom_stake_amount: {e}")
            return 0

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float,
                        time_in_force: str, current_time: datetime, entry_tag: Optional[str],
                        side: str, **kwargs) -> bool:
        try:
            # Get pair config
            pair_config = self.position_configs.get(pair, {
                'min_stake': 150.0,
                'min_amount': 0.1,
                'amount_precision': 3,
                'price_precision': 2
            })
            
            notional = amount * rate
            wallet_balance = self.wallets.get_free('USDT')
            
            logger.info(f"""
            Trade validation for {pair}:
            - Amount: {amount}
            - Rate: {rate}
            - Notional: {notional}
            - Min required: {pair_config['min_stake']}
            - Min amount: {pair_config['min_amount']}
            - Balance: {wallet_balance}
            """)
            
            # Validierungen
            if amount < pair_config['min_amount']:
                logger.info(f"Trade rejected - amount below minimum: {amount} < {pair_config['min_amount']}")
                return False
                
            if notional < pair_config['min_stake']:
                logger.info(f"Trade rejected - stake below minimum: {notional} < {pair_config['min_stake']}")
                return False
                
            if notional > wallet_balance * 0.99:
                logger.info(f"Trade rejected - insufficient balance: {notional} > {wallet_balance * 0.99}")
                return False
            
            logger.info(f"Trade confirmed for {pair}: {side} order at {rate}, amount: {amount}, notional: {notional}")
            return True
            
        except Exception as e:
            logger.error(f"Error in confirm_trade_entry: {e}")
            return False