#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_TrendV2"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily trend to 12h
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Donchian channel (20 periods)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            upper[i] = np.max(high[i-19:i+1])
            lower[i] = np.min(low[i-19:i+1])
    
    # Calculate volume confirmation (20-period average)
    vol_avg_20 = np.full(n, np.nan)
    for i in range(n):
        if i >= 19:
            vol_avg_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(vol_avg_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_confirmed = vol_current > 1.5 * vol_avg_20[i]  # Strong volume requirement
        
        if position == 0:
            # Long: break above upper Donchian with volume, above daily EMA50
            if price > upper[i] and vol_confirmed and price > ema50_1d_aligned[i]:
                signals[i] = 0.30
                position = 1
            # Short: break below lower Donchian with volume, below daily EMA50
            elif price < lower[i] and vol_confirmed and price < ema50_1d_aligned[i]:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: break below lower Donchian or trend change
            if price < lower[i] or price < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: break above upper Donchian or trend change
            if price > upper[i] or price > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals