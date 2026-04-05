#!/usr/bin/env python3
"""
exp_7498_1d_1w_donchian20_ema_volume_v1
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
In uptrend (price > weekly EMA50): buy breakout above 20-day high.
In downtrend (price < weekly EMA50): sell breakdown below 20-day low.
Volume filter ensures breakouts have participation. Targets 20-50 trades over 4 years (5-12/year).
Works in both bull and bear by following weekly trend direction.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7498_1d_1w_donchian20_ema_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
EMA_TREND = 50
VOLUME_MA = 20
VOLUME_THRESHOLD = 1.5  # 1.5x average volume
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=EMA_TREND, adjust=False, min_periods=EMA_TREND).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-day high/low)
    # Using rolling window with min_periods
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    donchian_low = low_series.rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA, min_periods=VOLUME_MA).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, EMA_TREND, VOLUME_MA, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(ema_1w_50_aligned[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                continue
        
        # Determine trend from weekly EMA50
        uptrend = close[i] > ema_1w_50_aligned[i]
        downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Volume confirmation
        high_volume = volume[i] > VOLUME_THRESHOLD * volume_ma[i] if not np.isnan(volume_ma[i]) else False
        
        # Entry conditions
        long_entry = (
            uptrend and                   # weekly uptrend
            close[i] > donchian_high[i] and  # break above 20-day high
            high_volume                   # volume confirmation
        )
        
        short_entry = (
            downtrend and                 # weekly downtrend
            close[i] < donchian_low[i] and  # break below 20-day low
            high_volume                   # volume confirmation
        )
        
        # Exit conditions - opposite breakout or trend change
        long_exit = (
            close[i] < donchian_low[i] or   # break below 20-day low
            not uptrend                     # trend change to downtrend
        )
        
        short_exit = (
            close[i] > donchian_high[i] or  # break above 20-day high
            not downtrend                   # trend change to uptrend
        )
        
        # Generate signals
        if position == 0:
            if long_entry:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
            elif short_entry:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = SIGNAL_SIZE
        elif position == -1:
            if short_exit:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -SIGNAL_SIZE
    
    return signals