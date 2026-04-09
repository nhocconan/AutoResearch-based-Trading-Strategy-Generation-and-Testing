#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA(50) trend filter + volume confirmation
# Donchian breakout captures strong momentum moves in both bull and bear markets
# 1w EMA(50) ensures we only trade in the direction of the higher timeframe trend
# Volume confirmation (current volume > 1.5 * 20-period average) filters weak breakouts
# Position size 0.25 to limit drawdown
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag
# Works in both bull/bear: trend filter adapts to market direction

name = "1d_1w_donchian_volume_trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1w data ONCE before loop for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(50) for trend filter
    close_1w = df_1w['close'].values
    ema_50 = np.full(len(df_1w), np.nan)
    multiplier = 2 / (50 + 1)
    ema_50[0] = close_1w[0]
    for i in range(1, len(df_1w)):
        ema_50[i] = (close_1w[i] * multiplier) + (ema_50[i-1] * (1 - multiplier))
    
    # Align 1w EMA to 1d timeframe
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate Donchian channels (20-period) on 1d
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 19:
            highest_high[i] = np.nan
            lowest_low[i] = np.nan
        else:
            start_idx = i - 19
            highest_high[i] = np.max(high[start_idx:i+1])
            lowest_low[i] = np.min(low[start_idx:i+1])
    
    # Calculate volume confirmation: current volume > 1.5 * 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(n):
        if i < 19:
            vol_ma_20[i] = np.nan
        else:
            start_idx = i - 19
            vol_ma_20[i] = np.mean(volume[start_idx:i+1])
    
    volume_ratio = np.full(n, np.nan)
    for i in range(n):
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] == 0:
            volume_ratio[i] = np.nan
        else:
            volume_ratio[i] = volume[i] / vol_ma_20[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d[i]) or
            np.isnan(volume_ratio[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from 1w EMA
        uptrend = close[i] > ema_50_1d[i]
        downtrend = close[i] < ema_50_1d[i]
        
        # Volume confirmation: strong volume supports breakout
        strong_volume = volume_ratio[i] > 1.5
        
        if position == 1:  # Long position
            # Exit conditions: price breaks below Donchian low OR trend reverses
            if close[i] <= lowest_low[i] or not uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions: price breaks above Donchian high OR trend reverses
            if close[i] >= highest_high[i] or not downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of 1w trend with volume confirmation
            bullish_breakout = close[i] > highest_high[i]
            bearish_breakout = close[i] < lowest_low[i]
            
            if bullish_breakout and uptrend and strong_volume:
                position = 1
                signals[i] = 0.25
            elif bearish_breakout and downtrend and strong_volume:
                position = -1
                signals[i] = -0.25
    
    return signals