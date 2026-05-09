#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_WeeklySupertrend_Breakout_DailyVolume"
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
    
    # Get weekly data for Supertrend
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 50:
        return np.zeros(n)
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR on weekly data
    high_w = df_w['high'].values
    low_w = df_w['low'].values
    close_w = df_w['close'].values
    
    # True Range
    tr1 = high_w[1:] - low_w[1:]
    tr2 = np.abs(high_w[1:] - close_w[:-1])
    tr3 = np.abs(low_w[1:] - close_w[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    if len(tr) >= atr_period:
        atr[atr_period] = np.nanmean(tr[1:atr_period+1])
        for i in range(atr_period + 1, len(tr)):
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Supertrend calculation
    hl2 = (high_w + low_w) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    supertrend = np.full_like(close_w, np.nan, dtype=np.float64)
    direction = np.full_like(close_w, 1, dtype=np.int8)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0] if not np.isnan(upper_band[0]) else np.nan
    direction[0] = 1
    
    for i in range(1, len(close_w)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = np.nan
            direction[i] = direction[i-1]
            continue
            
        if close_w[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_w[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend to daily timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_w, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_w, direction.astype(np.float64))
    
    # Daily volume confirmation: current volume > 1.5x 20-day average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or 
            np.isnan(direction_aligned[i]) or 
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Price above Supertrend (uptrend) with volume confirmation
            if close[i] > supertrend_aligned[i] and vol_ok and direction_aligned[i] == 1:
                signals[i] = 0.25
                position = 1
            # Short: Price below Supertrend (downtrend) with volume confirmation
            elif close[i] < supertrend_aligned[i] and vol_ok and direction_aligned[i] == -1:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price closes below Supertrend
            if close[i] < supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price closes above Supertrend
            if close[i] > supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals