#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Daily_Range_Swing_Rejection"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for range calculation (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Daily high, low, close for range calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Daily range and midpoint
    daily_range = high_1d - low_1d
    daily_mid = (high_1d + low_1d) / 2.0
    
    # Align daily range and midpoint to 4h timeframe
    daily_range_aligned = align_htf_to_ltf(prices, df_1d, daily_range)
    daily_mid_aligned = align_htf_to_ltf(prices, df_1d, daily_mid)
    
    # Previous day's range and midpoint (for rejection logic)
    prev_range = np.roll(daily_range_aligned, 1)
    prev_mid = np.roll(daily_mid_aligned, 1)
    prev_range[0] = np.nan
    prev_mid[0] = np.nan
    
    # 4h ATR for volatility filter
    tr = np.maximum(high[1:] - low[1:], np.absolute(high[1:] - close[:-1]))
    tr = np.maximum(tr, np.absolute(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(prev_range[i]) or np.isnan(prev_mid[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        rng = prev_range[i]
        mid = prev_mid[i]
        atr_val = atr[i]
        
        volume_confirmed = vol > 2.0 * vol_ma
        
        if position == 0:
            # Long rejection: price rejected below 50% of prev day's range, now reversing up with volume
            lower_bound = mid - 0.5 * rng
            if price > lower_bound and close[i-1] <= lower_bound and volume_confirmed:
                signals[i] = 0.25
                position = 1
            # Short rejection: price rejected above 50% of prev day's range, now reversing down with volume
            upper_bound = mid + 0.5 * rng
            if price < upper_bound and close[i-1] >= upper_bound and volume_confirmed:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price breaks below 25% of prev day's range or ATR stop
            lower_exit = mid - 0.75 * rng
            if price < lower_exit or price < close[i-1] - 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price breaks above 75% of prev day's range or ATR stop
            upper_exit = mid + 0.75 * rng
            if price > upper_exit or price > close[i-1] + 1.5 * atr_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals