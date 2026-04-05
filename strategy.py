#!/usr/bin/env python3
"""
exp_7518_1d_1w_donchian20_volume_v1
Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation.
Buy when price breaks above 20-day high with weekly uptrend and volume > 1.5x average.
Sell when price breaks below 20-day low with weekly downtrend and volume > 1.5x average.
Uses volatility-based position sizing and ATR stoploss to manage risk.
Targets 60-120 trades over 4 years (15-30/year) with strict breakout conditions.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7518_1d_1w_donchian20_volume_v1"
timeframe = "1d"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOLUME_MULTIPLIER = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_1w_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_50_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_50)
    
    # Calculate daily indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
    start = max(DONCHIAN_PERIOD, 20) + 1
    
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
        
        # Determine weekly trend
        weekly_uptrend = close[i] > ema_1w_50_aligned[i]
        weekly_downtrend = close[i] < ema_1w_50_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > VOLUME_MULTIPLIER * avg_volume[i]
        
        # Entry conditions
        long_entry = (
            close[i] > highest_high[i-1] and   # break above 20-day high
            weekly_uptrend and                 # weekly uptrend
            volume_confirm                     # volume confirmation
        )
        
        short_entry = (
            close[i] < lowest_low[i-1] and     # break below 20-day low
            weekly_downtrend and               # weekly downtrend
            volume_confirm                     # volume confirmation
        )
        
        # Exit conditions - mean reversion to mid-point
        mid_point = (highest_high[i] + lowest_low[i]) / 2
        long_exit = close[i] < mid_point
        short_exit = close[i] > mid_point
        
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