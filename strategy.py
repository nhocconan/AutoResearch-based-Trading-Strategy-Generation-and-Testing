#!/usr/bin/env python3
# 12h_trix_volume_sr_1d
# Hypothesis: TRIX momentum + volume spike + daily support/resistance. Long when TRIX crosses above zero with volume > 1.5x avg and price above daily pivot; short when TRIX crosses below zero with volume > 1.5x avg and price below daily pivot. Exit on TRIX reversal or price reaching opposite pivot. Designed to capture momentum bursts in both bull and bear markets with controlled trade frequency.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_trix_volume_sr_1d"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for pivot points (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot points (standard)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    
    # Align daily pivot to 12-hour chart
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Get 1-day data for TRIX (same timeframe as pivot for alignment)
    # TRIX: triple exponential smoothing of close, then % change
    close_1d_series = pd.Series(close_1d)
    ema1 = close_1d_series.ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = ema1.ewm(span=12, adjust=False, min_periods=12).mean()
    ema3 = ema2.ewm(span=12, adjust=False, min_periods=12).mean()
    trix_raw = 100 * (ema3 - ema3.shift(1)) / ema3.shift(1)
    trix = trix_raw.fillna(0).values  # TRIX indicator
    
    # Align TRIX to 12-hour chart
    trix_12h = align_htf_to_ltf(prices, df_1d, trix)
    
    # Volume confirmation: 24-period average (2 days of 12h data)
    avg_volume = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if data not available
        if np.isnan(pivot_12h[i]) or np.isnan(trix_12h[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: TRIX turns down or price reaches opposite pivot (below pivot)
            if trix_12h[i] < 0 or close[i] < pivot_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: TRIX turns up or price reaches opposite pivot (above pivot)
            if trix_12h[i] > 0 or close[i] > pivot_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.5x average volume
            volume_ok = volume[i] > 1.5 * avg_volume[i]
            
            # Long entry: TRIX crosses above zero with volume and price above pivot
            if i > 0 and trix_12h[i-1] <= 0 and trix_12h[i] > 0 and volume_ok and close[i] > pivot_12h[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: TRIX crosses below zero with volume and price below pivot
            elif i > 0 and trix_12h[i-1] >= 0 and trix_12h[i] < 0 and volume_ok and close[i] < pivot_12h[i]:
                position = -1
                signals[i] = -0.25
    
    return signals