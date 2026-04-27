#!/usr/bin/env python3
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
    
    # Get weekly data for calculations (called ONCE before loop)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 30:
        return np.zeros(n)
    
    # Calculate weekly close (for Donchian channel)
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    upper_weekly = np.full(len(close_weekly), np.nan)
    lower_weekly = np.full(len(close_weekly), np.nan)
    if len(close_weekly) >= 20:
        for i in range(20-1, len(close_weekly)):
            upper_weekly[i] = np.max(close_weekly[i-20+1:i+1])
            lower_weekly[i] = np.min(close_weekly[i-20+1:i+1])
    
    # Calculate weekly ATR (14-period) for volatility filter
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly_prev = np.roll(close_weekly, 1)
    close_weekly_prev[0] = close_weekly[0]
    
    tr1 = high_weekly - low_weekly
    tr2 = np.abs(high_weekly - close_weekly_prev)
    tr3 = np.abs(low_weekly - close_weekly_prev)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_14_weekly = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_14_weekly[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14_weekly[i] = (atr_14_weekly[i-1] * 13 + tr[i]) / 14
    
    # Align weekly indicators to daily timeframe
    upper_weekly_aligned = align_htf_to_ltf(prices, df_weekly, upper_weekly)
    lower_weekly_aligned = align_htf_to_ltf(prices, df_weekly, lower_weekly)
    atr_14_weekly_aligned = align_htf_to_ltf(prices, df_weekly, atr_14_weekly)
    
    # Calculate 20-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(20, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(upper_weekly_aligned[i]) or np.isnan(lower_weekly_aligned[i]) or 
            np.isnan(atr_14_weekly_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 2x average volume
        vol_filter = vol_ratio > 2.0
        
        if position == 0:
            # Long: Price breaks above weekly upper Donchian with volume
            if price > upper_weekly_aligned[i] and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price breaks below weekly lower Donchian with volume
            elif price < lower_weekly_aligned[i] and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below weekly lower Donchian or volatility spike (potential reversal)
            if price < lower_weekly_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above weekly upper Donchian or volatility spike (potential reversal)
            if price > upper_weekly_aligned[i] or (vol_ratio > 3.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_WeeklyDonchian_20_Volume"
timeframe = "1d"
leverage = 1.0