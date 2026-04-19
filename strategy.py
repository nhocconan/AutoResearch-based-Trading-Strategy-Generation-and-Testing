#!/usr/bin/env python3
# 12h_Donchian_20_Breakout_WeeklyTrend_Filter
# Hypothesis: 12h Donchian(20) breakout with weekly EMA trend filter and volume confirmation
# Donchian breakouts capture momentum from price channels; weekly EMA ensures alignment with higher timeframe trend
# Volume confirmation filters for institutional participation
# Designed for 12h timeframe to target 50-150 total trades over 4 years (12-37/year)
# Works in bull/bear via trend filter - only trade breakouts in direction of weekly trend

name = "12h_Donchian_20_Breakout_WeeklyTrend_Filter"
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
    
    # Weekly EMA for trend filter - calculated on 1w data (if available, fallback to 1d)
    try:
        df_1w = get_htf_data(prices, '1w')
        if len(df_1w) >= 50:
            trend_source = df_1w
            trend_close = df_1w['close'].values
        else:
            trend_source = get_htf_data(prices, '1d')
            trend_close = trend_source['close'].values
    except:
        trend_source = get_htf_data(prices, '1d')
        trend_close = trend_source['close'].values
    
    # Weekly EMA(50) for trend direction
    ema_50 = pd.Series(trend_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, trend_source, ema_50)
    
    # 12h data for Donchian channels and other indicators
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(df_12h['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_12h['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_12h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_12h, low_20)
    
    # Volume confirmation: volume > 1.3 * 30-period average
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_confirm = volume > (volume_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 30)  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(high_20_aligned[i]) or 
            np.isnan(low_20_aligned[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: only trade in direction of weekly EMA
        # Long when price above weekly EMA, short when below
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and uptrend
            if (close[i] > high_20_aligned[i] and 
                volume_confirm[i] and 
                uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and downtrend
            elif (close[i] < low_20_aligned[i] and 
                  volume_confirm[i] and 
                  downtrend):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price breaks below Donchian low or trend reverses
            if (close[i] < low_20_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price breaks above Donchian high or trend reverses
            if (close[i] > high_20_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals