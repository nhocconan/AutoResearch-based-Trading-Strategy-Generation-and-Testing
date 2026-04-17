#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation.
Long when price breaks above 20-period high with volume > 1.5x average and weekly close > weekly open.
Short when price breaks below 20-period low with volume > 1.5x average and weekly close < weekly open.
Exit when price reverts to 20-period midpoint or weekly trend reverses.
Uses 1w for trend filter, 6h for price/volume/DONCHIAN.
Target: 50-150 total trades over 4 years (12-37/year). Novel combination avoids saturated families.
"""

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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    # Weekly trend: 1 if bullish (close > open), -1 if bearish (close < open)
    weekly_trend = np.where(close_1w > open_1w, 1, -1)
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    
    # Calculate 6h Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        window_high = high[i-lookback+1:i+1]
        window_low = low[i-lookback+1:i+1]
        highest_high[i] = np.max(window_high)
        lowest_low[i] = np.min(window_low)
    
    # Donchian midpoint for exit
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(lookback-1, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(weekly_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        weekly_trend_val = weekly_trend_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        midpoint = donchian_mid[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume confirmation and bullish weekly trend
            if price > upper and vol_conf and weekly_trend_val == 1:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume confirmation and bearish weekly trend
            elif price < lower and vol_conf and weekly_trend_val == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to midpoint OR weekly trend turns bearish
            if price <= midpoint or weekly_trend_val == -1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to midpoint OR weekly trend turns bullish
            if price >= midpoint or weekly_trend_val == 1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0