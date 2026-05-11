#!/usr/bin/env python3
name = "1d_WeeklyTrend_WeeklyVolume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend and volume
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly trend: EMA21 > EMA55 = uptrend, EMA21 < EMA55 = downtrend
    ema21_1w = pd.Series(df_1w['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema55_1w = pd.Series(df_1w['close']).ewm(span=55, adjust=False, min_periods=55).mean().values
    trend_up_1w = ema21_1w > ema55_1w
    trend_down_1w = ema21_1w < ema55_1w
    
    # Weekly volume: current week volume > 1.5x 4-week average
    vol_mean_4w = pd.Series(df_1w['volume']).rolling(window=4, min_periods=4).mean().values
    vol_surge_1w = df_1w['volume'].values > 1.5 * vol_mean_4w
    
    # Align to daily
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    vol_surge_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_surge_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i]) or 
            np.isnan(vol_surge_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: weekly uptrend with volume surge
            if trend_up_1w_aligned[i] and vol_surge_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: weekly downtrend with volume surge
            elif trend_down_1w_aligned[i] and vol_surge_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: weekly trend changes to downtrend
            if trend_down_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: weekly trend changes to uptrend
            if trend_up_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals