#!/usr/bin/env python3
# 6h_1d_WickReversal_Volume_Filter
# Hypothesis: Fade long wicks (rejection) at daily highs/lows with volume confirmation.
# Uses weekly trend filter to avoid counter-trend trades. Works in bull/bear by only trading with weekly trend.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_WickReversal_Volume_Filter"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data ONCE before loop for high/low
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get weekly data ONCE before loop for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily high and low
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    close_1w_series = pd.Series(close_1w)
    ema34_1w = close_1w_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # 6h: Volume ratio (current vs 20-period average)
    volume = prices['volume'].values
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma20 > 0, vol_ma20, np.nan)
    
    # Align daily and weekly levels to 6h
    high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_1d)
    low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):  # Start after EMA and volume MA warmup
        # Get values
        close_val = prices['close'].iloc[i]
        high_val = prices['high'].iloc[i]
        low_val = prices['low'].iloc[i]
        high_1d_val = high_1d_aligned[i]
        low_1d_val = low_1d_aligned[i]
        ema34_1w_val = ema34_1w_aligned[i]
        vol_ratio_val = vol_ratio[i]
        
        # Skip if any value is NaN
        if (np.isnan(high_1d_val) or np.isnan(low_1d_val) or np.isnan(ema34_1w_val) or 
            np.isnan(vol_ratio_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Rejection of daily low (long lower wick) with volume, in weekly uptrend
            if (low_val <= low_1d_val and  # Touched or went below daily low
                close_val > low_1d_val and  # Closed back above daily low
                (close_val - low_val) > 0.6 * (high_val - low_val) and  # Long lower wick (>60% of range)
                vol_ratio_val > 1.5 and  # Volume confirmation
                close_val > ema34_1w_val):  # Only long in weekly uptrend
                signals[i] = 0.25
                position = 1
            # Short: Rejection of daily high (long upper wick) with volume, in weekly downtrend
            elif (high_val >= high_1d_val and  # Touched or went above daily high
                  close_val < high_1d_val and  # Closed back below daily high
                  (high_val - low_val) > 0 and
                  (high_val - close_val) > 0.6 * (high_val - low_val) and  # Long upper wick (>60% of range)
                  vol_ratio_val > 1.5 and  # Volume confirmation
                  close_val < ema34_1w_val):  # Only short in weekly downtrend
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Close back below daily low or weak close
            if close_val <= low_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Close back above daily high or weak close
            if close_val >= high_1d_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals