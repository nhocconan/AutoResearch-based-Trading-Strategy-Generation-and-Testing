#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_Trend_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels and EMA
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    weekly_high = df_weekly['high'].rolling(window=20, min_periods=20).max().values
    weekly_low = df_weekly['low'].rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    weekly_ema50 = pd.Series(df_weekly['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_weekly, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_weekly, weekly_low)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Daily volume filter: current volume > 1.5 * 20-day average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 50, 20)  # Donchian, EMA50, volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(weekly_ema50_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_high_val = weekly_high_aligned[i]
        weekly_low_val = weekly_low_aligned[i]
        weekly_ema50_val = weekly_ema50_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: daily close above weekly Donchian high + above weekly EMA50 + volume filter
            if close[i] > weekly_high_val and close[i] > weekly_ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: daily close below weekly Donchian low + below weekly EMA50 + volume filter
            elif close[i] < weekly_low_val and close[i] < weekly_ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: daily close below weekly EMA50
            if close[i] < weekly_ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: daily close above weekly EMA50
            if close[i] > weekly_ema50_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals