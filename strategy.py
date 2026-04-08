#!/usr/bin/env python3
# 12h_camarilla_pivot_1d_trend_volume
# Hypothesis: Camarilla pivot levels from 1-day data combined with EMA trend filter and volume confirmation.
# Long when price touches or breaks above Camarilla H3 level with uptrend (price > 1d EMA50) and volume > 1.3x average.
# Short when price touches or breaks below Camarilla L3 level with downtrend (price < 1d EMA50) and volume > 1.3x average.
# Exit when price returns to Camarilla P (pivot) level.
# Designed to capture mean-reversion bounces at key intraday levels in both ranging and trending markets.
# Target: 50-150 total trades over 4 years (~12-37/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume"
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
    
    # Get daily data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # P = (H + L + C) / 3
    # H3 = P + (H - L) * 1.1 / 2
    # L3 = P - (H - L) * 1.1 / 2
    pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_h3 = pivot + (high_1d - low_1d) * 1.1 / 2
    camarilla_l3 = pivot - (high_1d - low_1d) * 1.1 / 2
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price returns to or below pivot level
            if close[i] <= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to or above pivot level
            if close[i] >= pivot_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume confirmation: current volume > 1.3x average volume
            volume_ok = volume[i] > 1.3 * avg_volume[i]
            
            # Entry conditions: touch/break of Camarilla H3/L3 with trend alignment
            if (close[i] >= camarilla_h3_aligned[i]) and (close[i] > ema_50_1d_aligned[i]) and volume_ok:
                position = 1
                signals[i] = 0.25
            elif (close[i] <= camarilla_l3_aligned[i]) and (close[i] < ema_50_1d_aligned[i]) and volume_ok:
                position = -1
                signals[i] = -0.25
    
    return signals