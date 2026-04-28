#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 1d Williams %R extreme levels with volume confirmation and 1w EMA50 trend filter.
# Enter long when Williams %R(14) < -80 (oversold) with volume > 2.0x 20-bar average and price > 1w EMA50 (uptrend).
# Enter short when Williams %R(14) > -20 (overbought) with volume > 2.0x 20-bar average and price < 1w EMA50 (downtrend).
# Exit when Williams %R returns to neutral range (-50 to -50) or opposite extreme is reached.
# Uses discrete position sizing (0.25) to control risk. Target: 75-200 total trades over 4 years.
# Williams %R identifies exhaustion points, volume confirms reversal strength, 1w EMA50 filters counter-trend noise.
# Works in bull (buying dips in uptrend) and bear (selling rallies in downtrend) markets.

name = "4h_WilliamsR_1dExtreme_Volume_1wEMA50_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R parameters
    lookback = 14
    highest_high = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    williams_r = np.where(hl_range != 0, ((highest_high - close_1d) / hl_range) * -100, -50)
    
    # Align 1d Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 1w data for EMA50 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Calculate 1w EMA50
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 4h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R extreme conditions with volume confirmation and trend filter
        long_signal = williams_r_aligned[i] < -80 and volume_confirm[i] and close[i] > ema_50_1w_aligned[i]
        short_signal = williams_r_aligned[i] > -20 and volume_confirm[i] and close[i] < ema_50_1w_aligned[i]
        
        # Exit conditions: Williams %R returns to neutral range or opposite extreme
        long_exit = williams_r_aligned[i] > -50
        short_exit = williams_r_aligned[i] < -50
        
        # Handle entries and exits
        if long_signal and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_signal and position >= 0:
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