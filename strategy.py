#!/usr/bin/env python3
name = "1d_WeeklyDonchian20_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_weekly = df_weekly['close'].values
    ema_20_weekly = pd.Series(close_weekly).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    # Upper band: highest high of last 20 weekly periods
    upper_series = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 weekly periods
    lower_series = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    upper_aligned = align_htf_to_ltf(prices, df_weekly, upper_series)
    lower_aligned = align_htf_to_ltf(prices, df_weekly, lower_series)
    
    # Volume confirmation: current volume vs 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(ema_20_weekly_aligned[i]) or 
            np.isnan(upper_aligned[i]) or 
            np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly upper band, uptrend (price > weekly EMA20), volume confirmation
            if (close[i] > upper_aligned[i] and 
                close[i] > ema_20_weekly_aligned[i] and 
                volume_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly lower band, downtrend (price < weekly EMA20), volume confirmation
            elif (close[i] < lower_aligned[i] and 
                  close[i] < ema_20_weekly_aligned[i] and 
                  volume_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below weekly lower band (reversal signal)
            if close[i] < lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above weekly upper band (reversal signal)
            if close[i] > upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals