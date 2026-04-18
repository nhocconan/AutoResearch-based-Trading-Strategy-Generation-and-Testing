#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Supertrend calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate Supertrend on weekly data
    hl2_weekly = (df_weekly['high'].values + df_weekly['low'].values) / 2
    atr_period = 10
    atr_mult = 3.0
    
    # Calculate True Range and ATR
    tr1 = df_weekly['high'].values - df_weekly['low'].values
    tr2 = np.abs(df_weekly['high'].values - np.roll(df_weekly['close'].values, 1))
    tr3 = np.abs(df_weekly['low'].values - np.roll(df_weekly['close'].values, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    atr = np.full_like(tr, np.nan, dtype=np.float64)
    for i in range(atr_period, len(tr)):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Calculate Supertrend upper and lower bands
    upper_band = hl2_weekly + atr_mult * atr
    lower_band = hl2_weekly - atr_mult * atr
    
    # Initialize Supertrend
    supertrend = np.full_like(close, np.nan, dtype=np.float64)
    direction = np.ones_like(close, dtype=np.int8)  # 1 for uptrend, -1 for downtrend
    
    for i in range(atr_period, len(hl2_weekly)):
        if i == atr_period:
            supertrend[i] = lower_band[i]
            direction[i] = 1
        else:
            if close[i-1] > supertrend[i-1]:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
                direction[i] = 1
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
                direction[i] = -1
    
    # Align weekly Supertrend direction to 6h
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_weekly, direction.astype(np.float64))
    
    # Calculate 60-period EMA for trend confirmation
    close_series = pd.Series(close)
    ema60 = close_series.ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # Calculate volume spike detector (volume > 2x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(60, 20)  # need EMA60 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(supertrend_dir_aligned[i]) or np.isnan(ema60[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike confirmation
        vol_spike = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long entry: weekly uptrend, price above EMA60, with volume spike
            if (supertrend_dir_aligned[i] == 1 and 
                close[i] > ema60[i] and 
                vol_spike):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly downtrend, price below EMA60, with volume spike
            elif (supertrend_dir_aligned[i] == -1 and 
                  close[i] < ema60[i] and 
                  vol_spike):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly trend turns down or price crosses below EMA60
            if supertrend_dir_aligned[i] == -1 or close[i] < ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly trend turns up or price crosses above EMA60
            if supertrend_dir_aligned[i] == 1 or close[i] > ema60[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklySupertrend_EMA60_VolumeSpike"
timeframe = "6h"
leverage = 1.0