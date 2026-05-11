#!/usr/bin/env python3
name = "4h_Donchian_Breakout_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily and weekly data for trend filter
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 50 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Daily trend filter: EMA50 > EMA200
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(df_1d['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1d = ema50_1d > ema200_1d
    trend_down_1d = ema50_1d < ema200_1d
    
    # Weekly trend filter: EMA50 > EMA200
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1w = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    trend_up_1w = ema50_1w > ema200_1w
    trend_down_1w = ema50_1w < ema200_1w
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = np.zeros(n)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    # Align higher timeframe trends to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high in daily and weekly uptrend with volume surge
            if (close[i] > donchian_high[i] and 
                trend_up_1d_aligned[i] and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in daily and weekly downtrend with volume surge
            elif (close[i] < donchian_low[i] and 
                  trend_down_1d_aligned[i] and 
                  trend_down_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low or trend changes
            if (close[i] < donchian_low[i] or 
                not trend_up_1d_aligned[i] or 
                not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high or trend changes
            if (close[i] > donchian_high[i] or 
                not trend_down_1d_aligned[i] or 
                not trend_down_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals