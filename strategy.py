#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Breakout_Volume_Trend"
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
    
    # Get 1w data for trend and breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w high/low for breakout levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # 1w EMA34 for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w data to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Volume confirmation - 5-day average volume
    vol_ma = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get previous week's high/low (already aligned, so previous value is prior week's close)
        prev_high_1w = high_1w[i//7] if i >= 7 else np.nan  # Simplified: using index for weekly alignment
        prev_low_1w = low_1w[i//7] if i >= 7 else np.nan
        
        # Actually, we need to properly get the previous week's values from the aligned arrays
        # Since we don't have direct access to the weekly high/low arrays aligned, we'll use a different approach
        # Let's calculate the weekly high/low properly
        
    # Recalculate with proper alignment
    # Get weekly high and low series
    weekly_high = high_1w
    weekly_low = low_1w
    
    # Align them to daily
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above previous week's high + above 1w EMA34 + volume confirmation
            if (close[i] > weekly_high_aligned[i] and 
                close[i] > ema_34_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below previous week's low + below 1w EMA34 + volume confirmation
            elif (close[i] < weekly_low_aligned[i] and 
                  close[i] < ema_34_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below weekly low OR below 1w EMA34
            if close[i] < weekly_low_aligned[i] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above weekly high OR above 1w EMA34
            if close[i] > weekly_high_aligned[i] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals