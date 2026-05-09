#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklyDonchian20_1wEMA50_VolumeConfirm"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 50-period EMA on weekly close for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 20-period Donchian channels on daily
    highest_20 = np.full(n, np.nan)
    lowest_20 = np.full(n, np.nan)
    
    for i in range(20, n):
        highest_20[i] = np.max(high[i-20:i])
        lowest_20[i] = np.min(low[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Need 20 for Donchian, 50 for weekly EMA
    
    for i in range(start_idx, n):
        # Skip if required data unavailable (NaN from indicators)
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_1w = ema_50_1w_aligned[i]
        high_20 = highest_20[i]
        low_20 = lowest_20[i]
        vol = volume[i]
        
        # Calculate 20-period volume average for confirmation
        if i >= 20:
            vol_ma = np.mean(volume[i-20:i])
        else:
            vol_ma = np.mean(volume[:i]) if i > 0 else volume[i]
        
        if position == 0:
            # Enter long: High > 20-period high AND price > weekly EMA50 (uptrend) AND volume > 1.8x average
            if high[i] > high_20 and close[i] > ema_1w and vol > 1.8 * vol_ma:
                signals[i] = 0.25
                position = 1
            # Enter short: Low < 20-period low AND price < weekly EMA50 (downtrend) AND volume > 1.8x average
            elif low[i] < low_20 and close[i] < ema_1w and vol > 1.8 * vol_ma:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Low < 20-period low OR trend reverses (price < weekly EMA50)
            if low[i] < low_20 or close[i] < ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: High > 20-period high OR trend reverses (price > weekly EMA50)
            if high[i] > high_20 or close[i] > ema_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals