#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d volume confirmation and 1w EMA trend filter
# - Donchian breakout provides clear entry/exit levels based on price action
# - 1d volume surge confirms institutional participation in the breakout
# - 1w EMA filter ensures we only trade in the direction of the weekly trend
# - Designed for 60-120 total trades over 4 years (15-30/year) to minimize fee drag
# - Works in bull markets (breakouts continue) and bear markets (breakdowns continue)
name = "6h_Donchian20_1dVolume_1wEMA"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d volume confirmation (average volume)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_20_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_20_1d_avg = np.mean(vol_20_1d[-20:]) if len(vol_20_1d) >= 20 else np.mean(vol_20_1d[~np.isnan(vol_20_1d)])
    vol_20_1d_avg_arr = np.full(len(df_1d), vol_20_1d_avg)
    vol_confirm_1d = align_htf_to_ltf(prices, df_1d, vol_20_1d_avg_arr)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian(20) channels on 6h data
    def donchian_channels(high, low, window):
        upper = np.full_like(high, np.nan, dtype=float)
        lower = np.full_like(low, np.nan, dtype=float)
        for i in range(window-1, len(high)):
            upper[i] = np.max(high[i-window+1:i+1])
            lower[i] = np.min(low[i-window+1:i+1])
        return upper, lower
    
    upper, lower = donchian_channels(high, low, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_confirm_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian + volume surge + above weekly EMA
            if (close[i] > upper[i] and 
                volume[i] > vol_confirm_1d[i] * 1.5 and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian + volume surge + below weekly EMA
            elif (close[i] < lower[i] and 
                  volume[i] > vol_confirm_1d[i] * 1.5 and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below lower Donchian or weekly trend turns bearish
            if (close[i] < lower[i]) or (close[i] < ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above upper Donchian or weekly trend turns bullish
            if (close[i] > upper[i]) or (close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals