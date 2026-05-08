#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyBreakout_Volume_Trend_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter and breakout levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Upper band: highest high of last 20 weekly bars
    upper_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weekly bars
    lower_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly indicators to daily timeframe
    upper_1w_aligned = align_htf_to_ltf(prices, df_1w, upper_1w)
    lower_1w_aligned = align_htf_to_ltf(prices, df_1w, lower_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation - 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(upper_1w_aligned[i]) or np.isnan(lower_1w_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper band + above weekly EMA50 + volume confirmation
            if (close[i] > upper_1w_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower band + below weekly EMA50 + volume confirmation
            elif (close[i] < lower_1w_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price falls back below weekly lower band OR below weekly EMA50
            if close[i] < lower_1w_aligned[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price rises back above weekly upper band OR above weekly EMA50
            if close[i] > upper_1w_aligned[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals