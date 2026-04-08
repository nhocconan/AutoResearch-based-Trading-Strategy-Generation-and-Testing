#!/usr/bin/env python3
# 4h_daily_camarilla_pivot_volume_spike_v4
# Hypothesis: Camarilla pivot levels from 1d timeframe act as strong support/resistance on 4h.
# Long when price touches L3 level with volume spike (>1.5x avg volume) in bullish regime (close > 1d EMA50).
# Short when price touches H3 level with volume spike in bearish regime (close < 1d EMA50).
# Exit on opposite pivot touch or regime change.
# Uses 4h primary timeframe with 1d HTF for pivot levels and regime filter.
# Target: 75-200 total trades over 4 years to balance opportunity and fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_daily_camarilla_pivot_volume_spike_v4"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for spike detection (20-period)
    vol_s = pd.Series(volume)
    avg_volume = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for Camarilla pivots and regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    # Range = high - low
    # L3 = close - (high - low) * 1.1/4
    # H3 = close + (high - low) * 1.1/4
    range_1d = high_1d - low_1d
    L3 = close_1d - (range_1d * 1.1 / 4)
    H3 = close_1d + (range_1d * 1.1 / 4)
    
    # Calculate 1d EMA50 for regime filter
    close_1d_s = pd.Series(close_1d)
    ema50_1d = close_1d_s.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(L3_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_volume[i]) or
            np.isnan(close[i]) or np.isnan(high[i]) or np.isnan(low[i])):
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
            continue
        
        # Volume spike condition: current volume > 1.5x average volume
        volume_spike = volume[i] > 1.5 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit conditions:
            # 1. Price touches or crosses H3 level (short signal)
            # 2. Regime change: close < 1d EMA50 (bearish regime)
            if (high[i] >= H3_aligned[i] and volume_spike) or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions:
            # 1. Price touches or crosses L3 level (long signal)
            # 2. Regime change: close > 1d EMA50 (bullish regime)
            if (low[i] <= L3_aligned[i] and volume_spike) or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Price touches L3 level with volume spike AND bullish regime (close > 1d EMA50)
            if (low[i] <= L3_aligned[i] and volume_spike and 
                close[i] > ema50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Price touches H3 level with volume spike AND bearish regime (close < 1d EMA50)
            elif (high[i] >= H3_aligned[i] and volume_spike and 
                  close[i] < ema50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals