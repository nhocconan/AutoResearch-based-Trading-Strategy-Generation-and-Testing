#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with volume confirmation and weekly trend filter.
Long when price breaks above 20-day high with volume > 1.5x average and weekly close > weekly EMA50.
Short when price breaks below 20-day low with volume > 1.5x average and weekly close < weekly EMA50.
Exit when price reverts to 10-day MA or weekly trend reverses.
Uses 1d for price/volume/Donchian, 1w for trend filter.
Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 20-period Donchian channels on 1d
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Calculate 10-period MA for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Calculate weekly EMA50 for trend filter
    if len(close_1w) >= 50:
        ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        weekly_trend_up = close_1w > ema_50_1w
        weekly_trend_down = close_1w < ema_50_1w
    else:
        weekly_trend_up = np.zeros_like(close_1w, dtype=bool)
        weekly_trend_down = np.zeros_like(close_1w, dtype=bool)
        ema_50_1w = np.zeros_like(close_1w)
    
    # Align weekly indicators to 1d timeframe
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(lookback, 20)  # warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ma_10[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_conf = volume_confirm[i]
        weekly_up = weekly_trend_up_aligned[i]
        weekly_down = weekly_trend_down_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        ma_exit = ma_10[i]
        
        if position == 0:
            # Long: price breaks above upper channel with volume confirmation and weekly uptrend
            if price > upper_channel and vol_conf and weekly_up:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower channel with volume confirmation and weekly downtrend
            elif price < lower_channel and vol_conf and weekly_down:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to 10-day MA or weekly trend turns down
            if price < ma_exit or weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to 10-day MA or weekly trend turns up
            if price > ma_exit or weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Volume_WeeklyTrend"
timeframe = "1d"
leverage = 1.0