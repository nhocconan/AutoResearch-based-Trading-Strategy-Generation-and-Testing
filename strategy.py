#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels (from 1d) with volume confirmation and 12h trend filter
# Uses H3/L3 as entry points in trending markets (above/below 12h EMA), with exits at H4/L4
# Target: 20-40 trades per year (80-160 total over 4 years) for 4h timeframe
# Works in bull/bear by following 12h trend direction only

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_ = high_1d - low_1d
    
    # Camarilla levels
    h4 = close_1d + range_ * 1.1 / 2
    h3 = close_1d + range_ * 1.1 / 4
    l3 = close_1d - range_ * 1.1 / 4
    l4 = close_1d - range_ * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    
    # 12-hour data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # EMA(50) for 12h trend filter
    ema50_12h = np.zeros(len(close_12h))
    ema_multiplier = 2 / (50 + 1)
    ema50_12h[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        ema50_12h[i] = (close_12h[i] - ema50_12h[i-1]) * ema_multiplier + ema50_12h[i-1]
    
    # Align 12h EMA to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Average volume (20-period = 10 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is not ready
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        ema_trend = ema50_12h_aligned[i]
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        h4_val = h4_aligned[i]
        l4_val = l4_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average volume
        volume_confirm = vol > 1.5 * avg_vol
        
        if position == 0:
            # Long: Price crosses above H3 in uptrend (price > 12h EMA) + volume
            if (price > h3_val and price > ema_trend and volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price crosses below L3 in downtrend (price < 12h EMA) + volume
            elif (price < l3_val and price < ema_trend and volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price reaches H4 or trend reverses
            if (price >= h4_val or price < ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price reaches L4 or trend reverses
            if (price <= l4_val or price > ema_trend):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_Camarilla_Trend_Volume"
timeframe = "4h"
leverage = 1.0