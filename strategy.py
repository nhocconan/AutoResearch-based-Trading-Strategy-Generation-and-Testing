#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 12h Supertrend for trend direction and 1d Williams %R extremes for mean reversion entries.
# Supertrend (12h, ATR=10, mult=3) identifies the primary trend on 12h timeframe.
# Williams %R (1d, period=14) identifies overbought/oversold conditions: long when %R < -80, short when %R > -20.
# Volume confirmation (>1.5x 20-bar average) reduces false signals.
# This combines trend following with mean reversion in pullbacks, working in both bull and bear markets.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h (within proven winning range).
# Position size 0.25 balances return and drawdown. Discrete levels minimize fee churn.

name = "6h_Supertrend_12h_WilliamsR_1d_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h Supertrend (ATR=10, mult=3)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR(10)
    atr_10 = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + 3 * atr_10
    lower_band = hl2 - 3 * atr_10
    
    supertrend = np.zeros(len(close_12h))
    direction = np.ones(len(close_12h))  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1 and direction[i-1] == -1:
            supertrend[i] = lower_band[i]
        elif direction[i] == -1 and direction[i-1] == 1:
            supertrend[i] = upper_band[i]
        elif direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend and direction to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # Get daily data for Williams %R
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (period=14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close_1d) / denominator, -50)
    
    # Align Williams %R to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 6h volume spike: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h Supertrend
        uptrend = direction_aligned[i] == 1
        downtrend = direction_aligned[i] == -1
        
        # Williams %R extremes for mean reversion entries
        oversold = williams_r_aligned[i] < -80  # Long signal
        overbought = williams_r_aligned[i] > -20  # Short signal
        
        # Entry conditions with volume confirmation
        long_entry = uptrend and oversold and volume_spike[i]
        short_entry = downtrend and overbought and volume_spike[i]
        
        # Exit conditions: trend reversal or opposite Williams %R extreme
        long_exit = not uptrend or williams_r_aligned[i] > -50
        short_exit = not downtrend or williams_r_aligned[i] < -50
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals