#!/usr/bin/env python3
"""
Experiment #7894: 1-hour timeframe with 4h/1d trend filter and volume confirmation.
Hypothesis: In both bull and bear markets, price tends to continue in the direction of the 4h trend when breaking key levels with volume confirmation. Using 4h for trend direction (via EMA crossover) and 1d for regime filter (price vs EMA200) reduces false signals. 1h is used only for precise entry timing on pullbacks to the 4h EMA in the direction of the trend. Targets 60-150 trades over 4 years with low turnover to minimize fee drag.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7894_1h_4h_ema_trend_1d_regime_vol"
timeframe = "1h"
leverage = 1.0

# Parameters
EMA_FAST = 12
EMA_SLOW = 26
EMA_REGIME = 200
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.20
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 4h EMA crossover for trend direction
    close_4h = df_4h['close'].values
    ema_fast_4h = pd.Series(close_4h).ewm(span=EMA_FAST, adjust=False, min_periods=EMA_FAST).mean().values
    ema_slow_4h = pd.Series(close_4h).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    trend_4h = np.where(ema_fast_4h > ema_slow_4h, 1, -1)  # 1=uptrend, -1=downtrend
    trend_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_4h)
    
    # Calculate 1d regime filter: price vs EMA200
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=EMA_REGIME, adjust=False, min_periods=EMA_REGIME).mean().values
    regime_bull = close_1d > ema_200_1d  # True if bullish regime
    regime_bear = close_1d < ema_200_1d  # True if bearish regime
    regime_aligned = align_htf_to_ltf(prices, df_1d, regime_bull.astype(int))  # 1=bull, 0=bear
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1h EMA for entry timing (pullback to EMA)
    ema_1h = pd.Series(close).ewm(span=EMA_SLOW, adjust=False, min_periods=EMA_SLOW).mean().values
    
    # Volume confirmation
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(EMA_SLOW, EMA_REGIME, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if outside trading session
        if not in_session[i]:
            if position != 0:
                # Check stoploss even outside session
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Skip if HTF data not available
        if np.isnan(trend_4h_aligned[i]) or np.isnan(regime_aligned[i]):
            if position != 0:
                # Check stoploss
                if position == 1 and close[i] <= stop_price:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close[i] >= stop_price:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1 and close[i] <= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        elif position == -1 and close[i] >= stop_price:
            signals[i] = 0.0
            position = 0
            continue
        
        # Determine market bias
        bull_trend = trend_4h_aligned[i] == 1
        bear_trend = trend_4h_aligned[i] == -1
        bull_regime = regime_aligned[i] == 1
        bear_regime = regime_aligned[i] == 0
        
        # Volume confirmation
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions: pullback to 1h EMA in direction of trend
        pullback_long = close[i] <= ema_1h[i] * 1.001  # Allow small overshoot
        pullback_short = close[i] >= ema_1h[i] * 0.999
        
        # Long: bullish trend + bullish regime + pullback + volume
        long_entry = bull_trend and bull_regime and pullback_long and volume_confirmed
        # Short: bearish trend + bearish regime + pullback + volume
        short_entry = bear_trend and bear_regime and pullback_short and volume_confirmed
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals