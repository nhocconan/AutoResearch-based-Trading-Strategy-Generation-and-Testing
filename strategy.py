#!/usr/bin/env python3
"""
6h_Supertrend_Regime_Adaptive
Hypothesis: Combines Supertrend for trend direction with a choppy regime filter (Choppiness Index) to avoid whipsaws. In trending regimes (CHOP < 38.2), follow Supertrend signals. In choppy regimes (CHOP > 61.8), mean-revert at Supertrend extremes. Uses 1d HTF for regime filter to reduce noise. 6h timeframe targets 50-150 trades over 4 years. Works in bull/bear via trend following and in range via mean reversion at adaptive levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Supertrend calculation (6h)
    atr_period = 10
    multiplier = 3.0
    
    # TR and ATR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).ewm(span=atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Basic upper/lower bands
    hl2 = (high + low) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, n):
        if close[i] > upper_band[i-1]:
            direction[i] = 1
        elif close[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Choppiness Index (1d HTF for regime filter)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    chop_period = 14
    # True Range for 1d
    tr1_1d = high_1d - low_1d
    tr2_1d = np.abs(high_1d - np.roll(close_1d, 1))
    tr3_1d = np.abs(low_1d - np.roll(close_1d, 1))
    tr1_1d[0] = 0
    tr2_1d[0] = 0
    tr3_1d[0] = 0
    tr_1d = np.maximum(tr1_1d, np.maximum(tr2_1d, tr3_1d))
    atr_1d_sum = pd.Series(tr_1d).rolling(window=chop_period, min_periods=chop_period).sum().values
    
    max_high_1d = pd.Series(high_1d).rolling(window=chop_period, min_periods=chop_period).max().values
    min_low_1d = pd.Series(low_1d).rolling(window=chop_period, min_periods=chop_period).min().values
    
    # Avoid division by zero
    range_1d = max_high_1d - min_low_1d
    chop_1d = 100 * np.log10(atr_1d_sum / np.log10(chop_period)) / np.log10(range_1d)
    chop_1d = np.where(range_1d > 0, chop_1d, 50)  # default to neutral when range=0
    
    # Align 1d Choppiness Index to 6h
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need ATR (10), Supertrend needs ~2*ATR period, Chop (14), volume avg (20)
    start_idx = max(atr_period * 2, chop_period, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(supertrend[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        st_val = supertrend[i]
        chop_val = chop_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine regime: choppy or trending
            is_choppy = chop_val > 61.8
            is_trending = chop_val < 38.2
            
            if is_trending:
                # Trending regime: follow Supertrend direction
                if close_val > st_val and vol_conf:  # Uptrend
                    signals[i] = size
                    position = 1
                elif close_val < st_val and vol_conf:  # Downtrend
                    signals[i] = -size
                    position = -1
            elif is_choppy:
                # Choppy regime: mean reversion at Supertrend extremes
                # Long when price is significantly below Supertrend (oversold)
                # Short when price is significantly above Supertrend (overbought)
                deviation = (close_val - st_val) / st_val
                if deviation < -0.02 and vol_conf:  # 2% below ST -> long
                    signals[i] = size
                    position = 1
                elif deviation > 0.02 and vol_conf:  # 2% above ST -> short
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: price crosses below Supertrend OR regime shifts to choppy and mean reversion signal
            exit_trend = close_val < st_val
            exit_choppy_mean = (chop_val > 61.8) and ((close_val - st_val) / st_val > -0.01)  # Near ST
            
            if exit_trend or exit_choppy_mean:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Supertrend OR regime shifts to choppy and mean reversion signal
            exit_trend = close_val > st_val
            exit_choppy_mean = (chop_val > 61.8) and ((close_val - st_val) / st_val < 0.01)  # Near ST
            
            if exit_trend or exit_choppy_mean:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Supertrend_Regime_Adaptive"
timeframe = "6h"
leverage = 1.0