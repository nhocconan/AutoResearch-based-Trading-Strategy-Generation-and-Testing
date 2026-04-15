#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h 1-day Donchian Breakout with Volume Confirmation and ATR Stop
# Uses the previous 20-day high/low as support/resistance. Breakouts are traded only when
# confirmed by volume (1.5x 20-period median volume) and ATR-based stoploss.
# Works in bull markets (breakouts up) and bear markets (breakouts down). Target: 50-150 total trades.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data for 20-day Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-day high and low (rolling window)
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Previous day's 20-day high/low (shifted by 1 to avoid look-ahead)
    prev_high_20d = np.roll(high_20d, 1)
    prev_low_20d = np.roll(low_20d, 1)
    prev_high_20d[0] = np.nan
    prev_low_20d[0] = np.nan
    
    # Align to 4h timeframe
    prev_high_20d_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20d)
    prev_low_20d_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20d)
    
    # Calculate ATR (14-period) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25
    atr_multiplier = 2.0
    
    # Track entry price for stoploss
    entry_price = np.full(n, np.nan)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(prev_high_20d_aligned[i]) or np.isnan(prev_low_20d_aligned[i]) or
            np.isnan(atr_1d_aligned[i])):
            continue
        
        # Long entry: price breaks above previous 20-day high + volume confirmation
        if (close[i] > prev_high_20d_aligned[i] and
            volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
            position <= 0):
            position = 1
            signals[i] = base_size
            entry_price[i] = close[i]  # Record entry price at close of signal bar
        
        # Short entry: price breaks below previous 20-day low + volume confirmation
        elif (close[i] < prev_low_20d_aligned[i] and
              volume[i] > 1.5 * np.median(volume[max(0, i-20):i+1]) and
              position >= 0):
            position = -1
            signals[i] = -base_size
            entry_price[i] = close[i]
        
        # Exit conditions: stoploss or reverse breakout
        if position == 1:
            # Stoploss: price drops below entry - ATR*multiplier
            if not np.isnan(entry_price[i]) and close[i] < entry_price[i] - atr_multiplier * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Reverse signal: break below 20-day low
            elif close[i] < prev_low_20d_aligned[i]:
                position = 0
                signals[i] = 0.0
        
        elif position == -1:
            # Stoploss: price rises above entry + ATR*multiplier
            if not np.isnan(entry_price[i]) and close[i] > entry_price[i] + atr_multiplier * atr_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            # Reverse signal: break above 20-day high
            elif close[i] > prev_high_20d_aligned[i]:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "4h_1d_Donchian_Breakout_Volume_ATR"
timeframe = "4h"
leverage = 1.0