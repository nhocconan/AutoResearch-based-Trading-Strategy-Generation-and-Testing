#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_Camarilla_R1_S1_Breakout_Volume_T3_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot levels
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate previous 12h bar's Camarilla pivot levels
    prev_high = np.concatenate([[np.nan], high_12h[:-1]])
    prev_low = np.concatenate([[np.nan], low_12h[:-1]])
    prev_close = np.concatenate([[np.nan], close_12h[:-1]])
    
    pivot = (prev_high + prev_low + prev_close) / 3
    range_ = prev_high - prev_low
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    r2 = pivot + (range_ * 1.1 / 6)
    s2 = pivot - (range_ * 1.1 / 6)
    
    # Align 12h Camarilla levels to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    r2_aligned = align_htf_to_ltf(prices, df_12h, r2)
    s2_aligned = align_htf_to_ltf(prices, df_12h, s2)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # T3 filter: momentum filter to reduce whipsaw
    price_series = pd.Series(close)
    ema1 = price_series.ewm(span=3, adjust=False).values
    ema2 = ema1.ewm(span=3, adjust=False).values
    ema3 = ema2.ewm(span=3, adjust=False).values
    ema4 = ema3.ewm(span=3, adjust=False).values
    ema5 = ema4.ewm(span=3, adjust=False).values
    ema6 = ema5.ewm(span=3, adjust=False).values
    t3 = -(ema6 * 0.7) + (3 * (ema5 + ema4) * 0.6) - (3 * (ema3 + ema2) * 0.6) + ema1
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(r2_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(t3[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        
        # Volume filter
        volume_ok = vol > 1.5 * vol_ma
        
        # T3 momentum filter: only take long when T3 rising, short when falling
        t3_rising = t3[i] > t3[i-1]
        t3_falling = t3[i] < t3[i-1]
        
        if position == 0:
            # Long: price breaks above R1 with volume and T3 rising
            if price > r1_aligned[i] and volume_ok and t3_rising:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and T3 falling
            elif price < s1_aligned[i] and volume_ok and t3_falling:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: price returns to pivot or T3 turns down
            if price < pivot_aligned[i] or not t3_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price returns to pivot or T3 turns up
            if price > pivot_aligned[i] or not t3_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals