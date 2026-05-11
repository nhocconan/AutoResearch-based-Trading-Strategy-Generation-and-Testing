#!/usr/bin/env python3
name = "12h_Donchian20_1dTrend_VolumeBreakout_v3"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter (EMA50) and Donchian channels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Daily EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Daily Donchian channels (20-period)
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    upper_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # 12h volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(upper_1d_aligned[i]) or np.isnan(lower_1d_aligned[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above 1d Donchian upper + daily uptrend + volume confirmation
            if close[i] > upper_1d_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 1d Donchian lower + daily downtrend + volume confirmation
            elif close[i] < lower_1d_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below 1d Donchian lower OR daily trend turns down
            if close[i] < lower_1d_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above 1d Donchian upper OR daily trend turns up
            if close[i] > upper_1d_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals