#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume spike and daily trend filter.
# Camarilla levels (H4, L4) act as strong support/resistance in ranging markets.
# Volume spike confirms institutional interest at pivot levels.
# Daily trend filter ensures we trade in the direction of higher timeframe momentum.
# Target: 20-50 trades per year (80-200 over 4 years) for 4h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's range
    # H4 = C + 1.1*(H-L)/2, L4 = C - 1.1*(H-L)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_h4 = np.zeros(len(close_1d))
    camarilla_l4 = np.zeros(len(close_1d))
    
    for i in range(1, len(close_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        rang = prev_high - prev_low
        camarilla_h4[i] = prev_close + 1.1 * rang / 2
        camarilla_l4[i] = prev_close - 1.1 * rang / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Daily trend filter: 20-period EMA
    ema_20_1d = np.zeros(len(close_1d))
    ema_20_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        ema_20_1d[i] = 0.1 * close_1d[i] + 0.9 * ema_20_1d[i-1]
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Volume spike detector: current volume > 2.0 x 20-period average
    vol_avg = np.zeros(n)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(ema_20_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = vol_avg[i]
        h4_level = camarilla_h4_aligned[i]
        l4_level = camarilla_l4_aligned[i]
        daily_ema = ema_20_1d_aligned[i]
        
        # Volume spike: current volume > 2.0 x average volume
        volume_spike = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long setup: price touches L4 support + volume spike + price above daily EMA
            if (price <= l4_level * 1.002 and  # Allow small buffer for touch
                volume_spike and
                price > daily_ema):
                position = 1
                signals[i] = position_size
            # Short setup: price touches H4 resistance + volume spike + price below daily EMA
            elif (price >= h4_level * 0.998 and  # Allow small buffer for touch
                  volume_spike and
                  price < daily_ema):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches H4 resistance or volume drops significantly
            if (price >= h4_level * 0.995 or  # Slight penetration of H4
                vol < 0.3 * avg_vol):         # Significant volume drop
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches L4 support or volume drops significantly
            if (price <= l4_level * 1.005 or  # Slight penetration of L4
                vol < 0.3 * avg_vol):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Pivot_Volume_Spike_v1"
timeframe = "4h"
leverage = 1.0