#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullPower_BullTrend_Exit"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Elder Ray and trend
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 50:
        return np.zeros(n)
    
    # Daily high, low, close for Elder Ray calculation
    daily_high = df_d['high'].values
    daily_low = df_d['low'].values
    daily_close = df_d['close'].values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_series = pd.Series(daily_close)
    ema13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = daily_high - ema13  # High - EMA13
    bear_power = daily_low - ema13   # Low - EMA13 (negative values indicate bear pressure)
    
    # Align Elder Ray components to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_d, bear_power)
    
    # Daily EMA(34) for trend filter
    ema34_d = close_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_d_aligned = align_htf_to_ltf(prices, df_d, ema34_d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_ma20 = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema34_d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ok = volume[i] > 1.5 * vol_ma20[i]
        
        if position == 0:
            # Long: Bull Power > 0 (buying pressure) + price above EMA34 trend + volume confirmation
            if bull_power_aligned[i] > 0 and close[i] > ema34_d_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
        
        elif position == 1:
            # Exit long: Bear Power > 0 (selling pressure takes over) OR price breaks below EMA34
            if bear_power_aligned[i] > 0 or close[i] < ema34_d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals