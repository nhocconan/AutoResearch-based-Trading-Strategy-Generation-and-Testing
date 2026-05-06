#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extremes (80/20) with 1w Supertrend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold), 1w Supertrend = bullish, and volume > 2.0 * avg_volume(20)
# Short when 1d Williams %R > -20 (overbought), 1w Supertrend = bearish, and volume > 2.0 * avg_volume(20)
# Exit when 1d Williams %R crosses back through -50 (mean reversion to midpoint)
# Supertrend provides adaptive trend filtering with ATR-based stops, reducing whipsaws
# Williams %R extremes work in ranging markets; Supertrend ensures we trade with higher timeframe momentum
# Volume confirmation validates reversal strength while limiting overtrading
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Works in both bull (buy oversold dips) and bear (sell overbought rallies) markets

name = "12h_1dWilliamsR_Extreme_1wSupertrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least 14 completed 1d bars for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero (when high == low)
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    # Align 1d Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Get 1w data ONCE before loop for Supertrend calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:  # Need at least 10 completed weekly bars for ATR
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Supertrend (ATR=10, multiplier=3.0)
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # ATR
    atr_1w = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic Upper and Lower Bands
    hl2 = (high_1w + low_1w) / 2
    upper_band = hl2 + (multiplier * atr_1w)
    lower_band = hl2 - (multiplier * atr_1w)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_1w, np.nan, dtype=float)
    direction = np.full_like(close_1w, np.nan, dtype=float)  # 1 for uptrend, -1 for downtrend
    
    # Calculate Supertrend
    for i in range(1, len(close_1w)):
        if np.isnan(atr_1w[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = np.nan
            direction[i] = np.nan
        else:
            if i == 1:
                # Initialize first value
                supertrend[i] = upper_band[i]
                direction[i] = -1  # Start with downtrend assumption
            else:
                prev_supertrend = supertrend[i-1]
                prev_direction = direction[i-1]
                
                if close_1w[i-1] > prev_supertrend:
                    # Previous close was above previous Supertrend
                    if upper_band[i] < prev_supertrend:
                        upper_band[i] = prev_supertrend
                    if lower_band[i] > prev_supertrend:
                        lower_band[i] = prev_supertrend
                else:
                    # Previous close was below previous Supertrend
                    if upper_band[i] > prev_supertrend:
                        upper_band[i] = prev_supertrend
                    if lower_band[i] < prev_supertrend:
                        lower_band[i] = prev_supertrend
                
                # Determine current Supertrend and direction
                if close_1w[i] > upper_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1  # Uptrend
                elif close_1w[i] < lower_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1  # Downtrend
                else:
                    supertrend[i] = supertrend[i-1]
                    direction[i] = direction[i-1]
    
    # Align 1w Supertrend direction to 12h timeframe (wait for completed 1w bar)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_1w, direction)
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(supertrend_dir_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), 1w Supertrend bullish (1), volume spike, in session
            if (williams_r_aligned[i] < -80 and 
                supertrend_dir_aligned[i] == 1 and 
                volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), 1w Supertrend bearish (-1), volume spike, in session
            elif (williams_r_aligned[i] > -20 and 
                  supertrend_dir_aligned[i] == -1 and 
                  volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (mean reversion)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (mean reversion)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals