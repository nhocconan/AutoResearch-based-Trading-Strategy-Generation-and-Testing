#!/usr/bin/env python3
# 4h_4h_1d_Trend_Filter_With_4h_EMA21_and_Volume
# Hypothesis: Trend following on 4h using EMA21 for direction, filtered by 1d EMA50 trend for long-term bias.
# Volume confirmation (current volume > 1.5x 20-period average) ensures momentum behind moves.
# Long when: price > EMA21(4h) AND price > EMA50(1d) AND volume > 1.5x avg volume.
# Short when: price < EMA21(4h) AND price < EMA50(1d) AND volume > 1.5x avg volume.
# Uses EMA21 for responsive trend, EMA50 for filtering counter-trend noise.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "4h_4h_1d_Trend_Filter_With_4h_EMA21_and_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate EMA21 on 4h
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate average volume (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 50)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema21[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > EMA21 AND price > EMA50(1d) AND volume > 1.5x avg volume
            if (close[i] > ema21[i] and close[i] > ema50_1d_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < EMA21 AND price < EMA50(1d) AND volume > 1.5x avg volume
            elif (close[i] < ema21[i] and close[i] < ema50_1d_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price closes below EMA21 or trend weakens
            if close[i] < ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price closes above EMA21 or trend weakens
            if close[i] > ema21[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals