#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend and Volume Confirmation
Long: Price breaks above 20-day high + weekly close > weekly open + volume > 1.5x 20-day volume avg
Short: Price breaks below 20-day low + weekly close < weekly open + volume > 1.5x 20-day volume avg
Exit: Price retracement to 10-day EMA or opposite Donchian break
Uses daily Donchian channels, weekly trend filter, and volume confirmation
Target: 10-20 trades/year per symbol (40-80 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    
    # Calculate weekly trend: bullish if weekly close > weekly open
    weekly_open = df_weekly['open'].values
    weekly_close = df_weekly['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_weekly, weekly_bullish.astype(float))
    
    # Calculate daily Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 10-day EMA for exit
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 20-day volume average for volume filter
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian channels
    
    for i in range(start_idx, n):
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_10[i]) or np.isnan(vol_avg_20[i]) or
            np.isnan(weekly_bullish_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_avg = vol_avg_20[i]
        ema_val = ema_10[i]
        weekly_bull = weekly_bullish_aligned[i] > 0.5  # convert back to boolean
        
        if position == 0:
            # Long: Price breaks above 20-day high + weekly bullish + volume > 1.5x avg
            if price > high_20[i] and close[i-1] <= high_20[i] and weekly_bull and vol > 1.5 * vol_avg:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-day low + weekly bearish + volume > 1.5x avg
            elif price < low_20[i] and close[i-1] >= low_20[i] and not weekly_bull and vol > 1.5 * vol_avg:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retracement to 10-day EMA or breaks below 20-day low
            if price < ema_val or price < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retracement to 10-day EMA or breaks above 20-day high
            if price > ema_val or price > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0