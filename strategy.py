#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d + volume confirmation
# Uses weekly trend filter and daily Camarilla levels for mean reversion:
# - Long when price touches S3 level AND weekly trend is up AND volume > 20-period average
# - Short when price touches R3 level AND weekly trend is down AND volume > 20-period average
# - Exit when price crosses the daily pivot (midpoint) or weekly trend reverses
# - Designed for low frequency (target: 15-30 trades/year) to minimize fee drag
# - Camarilla levels provide precise support/resistance; weekly trend filter avoids counter-trend trades

name = "6h_camarilla_pivot_1w_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter (EMA 50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    # Camarilla formula: 
    # H4 = close + 1.5 * (high - low)
    # L4 = close - 1.5 * (high - low)
    # H3 = close + 1.125 * (high - low)
    # L3 = close - 1.125 * (high - low)
    # H2 = close + 0.75 * (high - low)
    # L2 = close - 0.75 * (high - low)
    # H1 = close + 0.5 * (high - low)
    # L1 = close - 0.5 * (high - low)
    # Pivot = (high + low + close) / 3
    
    # We need previous day's values, so shift by 1
    if len(high_1d) < 2:
        return np.zeros(n)
    
    prev_high = high_1d[:-1]  # yesterday's high
    prev_low = low_1d[:-1]    # yesterday's low
    prev_close = close_1d[:-1]  # yesterday's close
    
    # Calculate Camarilla levels for yesterday
    H3 = prev_close + 1.125 * (prev_high - prev_low)
    L3 = prev_close - 1.125 * (prev_high - prev_low)
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Align to 6h timeframe (these levels are valid for the entire day)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Weekly trend filter
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: current volume above average
        vol_confirm = volume[i] > vol_ma[i]
        
        # Price levels
        H3_val = H3_aligned[i]
        L3_val = L3_aligned[i]
        pivot_val = pivot_aligned[i]
        
        # Entry conditions with tolerance for touching levels
        # Use 0.1% tolerance for touching the levels
        tolerance = 0.001
        near_H3 = abs(high[i] - H3_val) / H3_val <= tolerance
        near_L3 = abs(low[i] - L3_val) / L3_val <= tolerance
        
        if position == 1:  # Long position
            # Exit when price crosses pivot or weekly trend turns down
            if close[i] < pivot_val or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit when price crosses pivot or weekly trend turns up
            if close[i] > pivot_val or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Long near L3 in uptrend with volume confirmation
            if near_L3 and uptrend and vol_confirm:
                position = 1
                signals[i] = 0.25
            # Short near H3 in downtrend with volume confirmation
            elif near_H3 and downtrend and vol_confirm:
                position = -1
                signals[i] = -0.25
    
    return signals