#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d weekly pivot levels (R1/S1) for breakout direction with volume confirmation.
# In ranging markets, price often respects daily pivot levels. Breakouts above R1 or below S1 with volume
# indicate institutional participation and continuation. Uses discrete position sizing (0.25) to limit drawdown.
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing strong moves.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Weekly Pivot Points (based on prior week) ===
    # Weekly high, low, close from Friday's data (using prior 5 trading days approx)
    # Simplified: use prior 1d high/low/close for daily pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily pivot: P = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # R1 = 2*P - L
    r1_1d = 2 * pivot_1d - low_1d
    # S1 = 2*P - H
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align to 6h timeframe (wait for prior day's close)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Volume filter: current volume > 2.0x 24-period volume SMA (4 days of 6h bars)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only (avoid Asian session noise)
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or
            np.isnan(s1_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # Break above R1 with volume
        if (close[i] > r1_1d_aligned[i]) and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # Break below S1 with volume
        elif (close[i] < s1_1d_aligned[i]) and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Pivot_R1_S1_Breakout_Volume_v1"
timeframe = "6h"
leverage = 1.0