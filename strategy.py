#!/usr/bin/env python3
name = "1d_Donchian_20_Breakout_1wTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA50 trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_prev = np.roll(ema50_1w, 1)
    ema50_1w_prev[0] = np.nan
    trend_up_1w = ema50_1w > ema50_1w_prev
    trend_down_1w = ema50_1w < ema50_1w_prev
    
    # Align weekly trend to daily
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # Daily Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
        else:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: current volume > 1.5x 20-day average
    vol_ma20 = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1]) if i > 0 else 0
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high in weekly uptrend with volume surge
            if (close[i] > donchian_high[i] and 
                trend_up_1w_aligned[i] and 
                volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low in weekly downtrend with volume surge
            elif (close[i] < donchian_low[i] and 
                  trend_down_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_ma20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Donchian low or weekly trend changes
            if (close[i] < donchian_low[i] or not trend_up_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above Donchian high or weekly trend changes
            if (close[i] > donchian_high[i] or not trend_down_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals