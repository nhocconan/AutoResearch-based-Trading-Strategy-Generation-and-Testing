#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_Weekly_Channel_Breakout_With_Volume_Confirmation"
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
    
    # Get weekly data for Donchian channel calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly Donchian channel (20-period)
    high_20w = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20w = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # Align weekly channel to daily timeframe
    high_20w_aligned = align_htf_to_ltf(prices, df_1w, high_20w)
    low_20w_aligned = align_htf_to_ltf(prices, df_1w, low_20w)
    
    # Daily EMA50 for trend filter
    ema_50d = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Daily volume spike detection
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_20w_aligned[i]) or np.isnan(low_20w_aligned[i]) or 
            np.isnan(ema_50d[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 2.0 * vol_ma20[i]  # Require volume spike
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high with uptrend and volume spike
            if close[i] > high_20w_aligned[i] and close[i] > ema_50d[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low with downtrend and volume spike
            elif close[i] < low_20w_aligned[i] and close[i] < ema_50d[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price falls below weekly Donchian low or trend turns down
            if close[i] < low_20w_aligned[i] or close[i] < ema_50d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price rises above weekly Donchian high or trend turns up
            if close[i] > high_20w_aligned[i] or close[i] > ema_50d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals