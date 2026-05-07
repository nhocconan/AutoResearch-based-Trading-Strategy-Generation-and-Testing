#!/usr/bin/env python3
# 6h_Trix_WeeklyTrend_VolumeSpike
# Hypothesis: TRIX on 6h filters noise and identifies momentum; weekly trend (from 1w close > EMA50) provides directional bias; volume spikes confirm institutional participation. Works in bull/bear by aligning with weekly trend. Target: 20-40 trades/year with strict entry conditions to minimize fee drag.

timeframe = "6h"
name = "6h_Trix_WeeklyTrend_VolumeSpike"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) == 0:
        return np.zeros(n)
    
    # Calculate EMA50 on weekly closes
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = df_1w['close'].values > ema_50_1w
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    
    # Get daily data for additional context (optional, can be removed if not needed)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate TRIX on 6h close (1-period EMA triple, then ROC)
    # TRIX = EMA(EMA(EMA(close), period), period), period) then % change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change(periods=1) * 100  # Convert to percentage
    trix_values = trix.values
    
    # Volume spike: 2.5x average volume (50-period = ~2 days on 6h chart)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 15)  # Ensure we have TRIX and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(trix_values[i]) or np.isnan(weekly_uptrend_aligned[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: TRIX turning up (>0) with volume spike and weekly uptrend
            if trix_values[i] > 0 and trix_values[i] > trix_values[i-1] and volume[i] > 2.5 * vol_ma[i] and weekly_uptrend_aligned[i] > 0.5:
                signals[i] = 0.25
                position = 1
            # Short: TRIX turning down (<0) with volume spike and weekly downtrend
            elif trix_values[i] < 0 and trix_values[i] < trix_values[i-1] and volume[i] > 2.5 * vol_ma[i] and weekly_uptrend_aligned[i] < 0.5:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: TRIX turns negative or weekly trend fails
            if trix_values[i] < 0 or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: TRIX turns positive or weekly trend fails
            if trix_values[i] > 0 or weekly_uptrend_aligned[i] < 0.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals