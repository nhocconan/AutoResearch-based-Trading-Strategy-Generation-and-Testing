#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_Volume_Regime
Camarilla pivot breakout on 12h with volume confirmation and 1d chop regime filter.
- Calculate Camarilla levels from prior 1d OHLC
- Long when close breaks above R1 with volume > 1.5x 20-period average
- Short when close breaks below S1 with volume > 1.5x 20-period average
- Use 1d chop regime: only trade when CHOP(14) < 61.8 (trending market)
- Fixed position size: 0.25
- Designed for 12-25 trades/year per symbol
Works in bull (breaks up) and bear (breaks down) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def calculate_chop(high, low, close, window=14):
    """Calculate Choppiness Index."""
    atr = np.zeros_like(high)
    for i in range(1, len(high)):
        atr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    # True Range sum over window
    tr_sum = np.zeros_like(high)
    for i in range(window, len(high)):
        tr_sum[i] = np.sum(atr[i-window+1:i+1])
    
    # Highest high and lowest low over window
    max_high = np.zeros_like(high)
    min_low = np.zeros_like(high)
    for i in range(window-1, len(high)):
        max_high[i] = np.max(high[i-window+1:i+1])
        min_low[i] = np.min(low[i-window+1:i+1])
    
    # Chop calculation
    chop = np.full_like(high, 50.0, dtype=float)
    for i in range(window, len(high)):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (max_high[i] - min_low[i])) / np.log10(window)
        else:
            chop[i] = 50.0
    return chop

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla calculation and chop filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Chop for regime filter
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, window=14)
    chop_1d_12h = align_ltf_to_hlf(prices, df_1d, chop_1d)  # align 1d chop to 12h
    
    # Calculate volume average (20-period)
    vol_ma = np.zeros(n)
    vol_sum = 0
    for i in range(n):
        vol_sum += prices['volume'].iloc[i]
        if i >= 20:
            vol_sum -= prices['volume'].iloc[i-20]
        if i >= 19:
            vol_ma[i] = vol_sum / 20
        else:
            vol_ma[i] = np.nan
    
    signals = np.zeros(n)
    
    start_idx = 20  # need volume MA and chop
    
    for i in range(start_idx, n):
        # Get prior 1d OHLC for Camarilla (yesterday's data)
        # Find index of prior 1d bar in 1d data
        # Since we're on 12h timeframe, we need the most recent completed 1d bar
        # We'll use the 1d data index that corresponds to prior day
        
        # Simple approach: use prior 1d bar's OHLC
        # We need to ensure we're using completed 1d bar
        # For 12h timeframe, we can use the 1d data that ended at least 12h ago
        
        # Calculate Camarilla levels from prior 1d bar
        # We'll use a rolling window approach on 1d data
        if i < 2:  # need at least 2 12h bars to get prior day
            continue
            
        # Get index in 1d data for prior day
        # This is approximate - we'll use the last available 1d bar
        idx_1d = min(len(df_1d) - 1, i // 2)  # 2x 12h bars per day
        if idx_1d < 1:
            continue
            
        # Prior 1d OHLC (yesterday's completed bar)
        phigh = high_1d[idx_1d - 1]
        plow = low_1d[idx_1d - 1]
        pclose = close_1d[idx_1d - 1]
        
        # Calculate Camarilla levels
        range_val = phigh - plow
        if range_val <= 0:
            continue
            
        R1 = pclose + (range_val * 1.1 / 12)
        S1 = pclose - (range_val * 1.1 / 12)
        
        # Current 12h bar data
        close_price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Check regime: only trade in trending markets (chop < 61.8)
        if np.isnan(chop_1d_12h[i]) or chop_1d_12h[i] >= 61.8:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: volume > 1.5x 20-period average
        if np.isnan(vol_ma[i]) or volume <= 1.5 * vol_ma[i]:
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        if close_price > R1:
            signals[i] = 0.25  # long
        elif close_price < S1:
            signals[i] = -0.25  # short
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0