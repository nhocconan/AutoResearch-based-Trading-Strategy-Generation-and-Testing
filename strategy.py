#!/usr/bin/env python3
"""
6h_Pivot_Reversal_Daily_Trend
Hypothesis: Mean-reversion at daily pivot point with 1d EMA34 trend filter on 6b timeframe.
Buys near pivot in uptrend, sells near pivot in downtrend. Designed to capture 
mean-reversion in both bull and bear markets by following higher timeframe trend.
Target: 20-50 trades/year to minimize fee drag.
"""
name = "6h_Pivot_Reversal_Daily_Trend"
timeframe = "6h"
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
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily pivot point (PP) from previous day
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 6h timeframe
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 1.5 * 24-period average
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price near pivot (+/- 0.1%) + 1d uptrend + volume filter
            near_pivot = abs(close[i] - pp_aligned[i]) / pp_aligned[i] < 0.001
            if near_pivot and close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price near pivot (+/- 0.1%) + 1d downtrend + volume filter
            elif near_pivot and close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price moves 0.5% away from pivot in opposite direction
            if position == 1:
                if close[i] < pp_aligned[i] * 0.995:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pp_aligned[i] * 1.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals