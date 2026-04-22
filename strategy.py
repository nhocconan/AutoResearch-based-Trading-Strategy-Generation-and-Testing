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
    
    # Load daily data - ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily pivot points from previous day
    prev_high = df_daily['high'].shift(1).values
    prev_low = df_daily['low'].shift(1).values
    prev_close = df_daily['close'].shift(1).values
    
    # Classic pivot points
    pivot = (prev_high + prev_low + prev_close) / 3
    r1 = 2 * pivot - prev_low
    s1 = 2 * pivot - prev_high
    r2 = pivot + (prev_high - prev_low)
    s2 = pivot - (prev_high - prev_low)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_daily, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_daily, r1)
    s1_aligned = align_htf_to_ltf(prices, df_daily, s1)
    r2_aligned = align_htf_to_ltf(prices, df_daily, r2)
    s2_aligned = align_htf_to_ltf(prices, df_daily, s2)
    
    # Calculate 60-period EMA on 6h for trend filter
    close_series = pd.Series(close)
    ema_60 = close_series.ewm(span=60, min_periods=60, adjust=False).mean().values
    
    # Calculate volume spike detector
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.where(vol_ma_20 > 0, vol_ma_20, 1)
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(r2_aligned[i]) or 
            np.isnan(s2_aligned[i]) or np.isnan(ema_60[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price above S1, EMA up, volume spike
            if (close[i] > s1_aligned[i] and 
                close[i] > ema_60[i] and 
                vol_ratio[i] > 2.0):
                signals[i] = 0.25
                position = 1
            # Short conditions: price below R1, EMA down, volume spike
            elif (close[i] < r1_aligned[i] and 
                  close[i] < ema_60[i] and 
                  vol_ratio[i] > 2.0):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price reaches opposite pivot level or trend reversal
            if position == 1:
                if close[i] < pivot_aligned[i] or close[i] < ema_60[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > pivot_aligned[i] or close[i] > ema_60[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6H_Pivot_R1S1_Volume_EMA_Filter"
timeframe = "6h"
leverage = 1.0