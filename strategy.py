#!/usr/bin/env python3
# 12h_1d_1w_pivot_volume_reversal_v1
# Hypothesis: Trade reversals at weekly Camarilla pivot levels with daily volume confirmation on 12h timeframe.
# Uses weekly Camarilla pivot levels (H4/L4) as strong support/resistance.
# Long when price touches weekly L4 with daily volume surge and price above weekly EMA50 (uptrend).
# Short when price touches weekly H4 with daily volume surge and price below weekly EMA50 (downtrend).
# Designed for 12h timeframe to target 12-37 trades/year (50-150 total over 4 years).
# Weekly pivot levels provide institutional support/resistance, working in both bull and bear markets.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_pivot_volume_reversal_v1"
timeframe = "12h"
leverage = 1.0

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close arrays"""
    n = len(high)
    H4 = np.full(n, np.nan)
    L4 = np.full(n, np.nan)
    
    for i in range(n):
        if not (np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i])):
            diff = high[i] - low[i]
            H4[i] = close[i] + (diff * 1.1 / 2)
            L4[i] = close[i] - (diff * 1.1 / 2)
    
    return H4, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly data for Camarilla pivots and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    H4_1w, L4_1w = calculate_camarilla_pivots(high_1w, low_1w, close_1w)
    
    # EMA50 for weekly trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to 12h timeframe
    H4_1w_aligned = align_htf_to_ltf(prices, df_1w, H4_1w)
    L4_1w_aligned = align_htf_to_ltf(prices, df_1w, L4_1w)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume confirmation: volume > 2x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 50  # Ensure EMA50 is ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(H4_1w_aligned[i]) or np.isnan(L4_1w_aligned[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume surge condition
        vol_surge = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        # Price tolerance for pivot touch (0.1% of price)
        tol = close[i] * 0.001
        
        if position == 1:  # Long position
            # Exit: price moves above weekly EMA50 (take profit) or touches H4 (reverse)
            if close[i] > ema50_1w_aligned[i] * 1.02 or close[i] >= H4_1w_aligned[i] - tol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves below weekly EMA50 (take profit) or touches L4 (reverse)
            if close[i] < ema50_1w_aligned[i] * 0.98 or close[i] <= L4_1w_aligned[i] + tol:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches weekly L4 with volume surge and price below weekly EMA50
            if (abs(close[i] - L4_1w_aligned[i]) <= tol and vol_surge and 
                close[i] < ema50_1w_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches weekly H4 with volume surge and price above weekly EMA50
            elif (abs(close[i] - H4_1w_aligned[i]) <= tol and vol_surge and 
                  close[i] > ema50_1w_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals