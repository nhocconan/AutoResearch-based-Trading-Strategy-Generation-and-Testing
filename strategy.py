#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian_Breakout_v3"
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
    
    # Get weekly data for Donchian channel calculation
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 20:
        return np.zeros(n)
    
    # Calculate 20-week Donchian channel
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    
    # Calculate rolling max/min for Donchian (20-period)
    high_series = pd.Series(high_w)
    low_series = pd.Series(low_w)
    upper_w = high_series.rolling(window=20, min_periods=20).max().values
    lower_w = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian levels to daily timeframe
    upper_w_aligned = align_htf_to_ltf(prices, df_w, upper_w)
    lower_w_aligned = align_htf_to_ltf(prices, df_w, lower_w)
    
    # Weekly trend filter: 50-week EMA on weekly close
    close_w = df_w['close'].values
    ema50_w = pd.Series(close_w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_w_aligned = align_htf_to_ltf(prices, df_w, ema50_w)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for weekly EMA50
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if (np.isnan(upper_w_aligned[i]) or 
            np.isnan(lower_w_aligned[i]) or
            np.isnan(ema50_w_aligned[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        upper_val = upper_w_aligned[i]
        lower_val = lower_w_aligned[i]
        ema50_val = ema50_w_aligned[i]
        vol_filter = volume_filter[i]
        
        if position == 0:
            # Enter long: Price breaks above weekly Donchian upper + above EMA50 + volume filter
            if close[i] > upper_val and close[i] > ema50_val and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Price breaks below weekly Donchian lower + below EMA50 + volume filter
            elif close[i] < lower_val and close[i] < ema50_val and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly Donchian lower
            if close[i] < lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly Donchian upper
            if close[i] > upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals