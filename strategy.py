# Your trading strategy code here
#!/usr/bin/env python3
"""
Weekly Donchian Breakout with Daily Trend Filter - 1d Strategy
Captures weekly trend continuations using Donchian breakouts filtered by daily EMA.
Designed for low trade frequency to minimize fee drag in both bull and bear markets.
"""
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
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian(20) channels
    highest_high_weekly = np.full(len(close_weekly), np.nan)
    lowest_low_weekly = np.full(len(close_weekly), np.nan)
    for i in range(20, len(close_weekly)):
        highest_high_weekly[i] = np.max(high_weekly[i-20:i])
        lowest_low_weekly[i] = np.min(low_weekly[i-20:i])
    
    # Align weekly Donchian levels to daily timeframe
    highest_high_aligned = align_htf_to_ltf(prices, df_weekly, highest_high_weekly)
    lowest_low_aligned = align_htf_to_ltf(prices, df_weekly, lowest_low_weekly)
    
    # Get daily EMA(50) for trend filter
    ema_50_daily = np.full(n, np.nan)
    if n >= 50:
        ema_50_daily[49] = np.mean(close[:50])
        for i in range(50, n):
            ema_50_daily[i] = (close[i] * 2/51) + (ema_50_daily[i-1] * 49/51)
    
    # Calculate daily ATR(14) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # need EMA, Donchian, ATR, volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(ema_50_daily[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below daily EMA50
        trend_up = close[i] > ema_50_daily[i]
        trend_down = close[i] < ema_50_daily[i]
        
        if position == 0:
            # Long entry: close above weekly Donchian upper + 0.3*ATR, with volume and trend filter
            if (close[i] > highest_high_aligned[i] + 0.3 * atr[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below weekly Donchian lower - 0.3*ATR, with volume and trend filter
            elif (close[i] < lowest_low_aligned[i] - 0.3 * atr[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below weekly Donchian lower or ATR-based stop
            if close[i] < lowest_low_aligned[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above weekly Donchian upper or ATR-based stop
            if close[i] > highest_high_aligned[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "Weekly_Donchian20_DailyEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0