#!/usr/bin/env python3
# 12h_ema_cross_volume_filter_v1
# Hypothesis: Uses EMA crossover (9/21) on 12h with volume confirmation and 1d trend filter.
# Goes long when EMA9 crosses above EMA21 in uptrend (price > 1d EMA50) with volume surge.
# Goes short when EMA9 crosses below EMA21 in downtrend (price < 1d EMA50) with volume surge.
# Designed for low trade frequency (15-30/year) to avoid fee drag, works in bull/bear via trend filter.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_ema_cross_volume_filter_v1"
timeframe = "12h"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily trend filter: EMA50
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 12h EMA crossover
    ema9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume confirmation
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema9[i]) or np.isnan(ema21[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        daily_uptrend = close[i] > ema50_1d_aligned[i]
        daily_downtrend = close[i] < ema50_1d_aligned[i]
        
        # EMA crossover signals
        ema9_above_ema21 = ema9[i] > ema21[i]
        ema9_below_ema21 = ema9[i] < ema21[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.8 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: EMA cross down or trend change
            if ema9_below_ema21 or not daily_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA cross up or trend change
            if ema9_above_ema21 or not daily_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: EMA9 crosses above EMA21 in uptrend
                if daily_uptrend and ema9_above_ema21 and ema9[i-1] <= ema21[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: EMA9 crosses below EMA21 in downtrend
                elif daily_downtrend and ema9_below_ema21 and ema9[i-1] >= ema21[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals