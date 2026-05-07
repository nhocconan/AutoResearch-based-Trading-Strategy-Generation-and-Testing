#!/usr/bin/env python3
"""
6h_Weekly_Pivot_Pullback_1dTrend_Volume
Hypothesis: Price pullback to weekly pivot (PP) on 6h timeframe with 1d EMA34 trend filter and volume spike (2x average) captures mean reversion in both bull and bear markets. Weekly PP acts as institutional support/resistance. Trend filter ensures we trade with higher timeframe momentum, reducing whipsaw in sideways markets. Designed for low trade frequency (~20-40/year) to avoid fee drag.
"""
name = "6h_Weekly_Pivot_Pullback_1dTrend_Volume"
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
    
    # Get weekly data for pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate weekly pivot point from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Weekly pivot point (PP)
    weekly_pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1w, weekly_pp)
    
    # Get daily data for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume filter: current volume > 2.0 * 24-period average (stricter)
    vol_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need sufficient warmup for averages
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(weekly_pp_aligned[i]) or np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_avg[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price pulls back to weekly PP + 1d uptrend + volume spike
            if close[i] <= weekly_pp_aligned[i] * 1.005 and close[i] >= weekly_pp_aligned[i] * 0.995 and \
               close[i] > ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price pulls back to weekly PP + 1d downtrend + volume spike
            elif close[i] <= weekly_pp_aligned[i] * 1.005 and close[i] >= weekly_pp_aligned[i] * 0.995 and \
                 close[i] < ema_34_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price moves 0.5% away from weekly PP in favor of trend
            if position == 1:  # Long
                if close[i] >= weekly_pp_aligned[i] * 1.005:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (Short)
                if close[i] <= weekly_pp_aligned[i] * 0.995:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals