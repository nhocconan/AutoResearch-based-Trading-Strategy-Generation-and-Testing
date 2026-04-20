#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with weekly trend filter and volume confirmation
# - Long: Price breaks above R1 (resistance 1) from daily Camarilla + price above weekly EMA50 + volume > 1.5x avg
# - Short: Price breaks below S1 (support 1) + price below weekly EMA50 + volume > 1.5x avg
# - Exit: Price crosses back below R1 (long) or above S1 (short) OR price crosses weekly EMA50 in opposite direction
# - Weekly EMA50 filters trend direction to avoid counter-trend trades
# - Camarilla levels provide institutional support/resistance
# - Volume confirmation reduces false breakouts
# - Target: 15-30 trades per year per symbol (60-120 total over 4 years)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_6h = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate daily Camarilla levels
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Range = H - L
    range_1d = high_1d - low_1d
    # Camarilla levels
    r1_1d = pivot_1d + (range_1d * 1.1 / 12)  # Resistance 1
    s1_1d = pivot_1d - (range_1d * 1.1 / 12)  # Support 1
    
    # Align daily levels to 6h
    r1_1d_6h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_6h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # 6h price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):  # Start after warmup
        # Skip if NaN in critical values
        if (np.isnan(r1_1d_6h[i]) or np.isnan(s1_1d_6h[i]) or 
            np.isnan(ema_50_1w_6h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        
        if position == 0:
            # Long entry: price breaks above R1 + above weekly EMA50 + volume surge
            if (price > r1_1d_6h[i] and price > ema_50_1w_6h[i] and 
                vol > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short entry: price breaks below S1 + below weekly EMA50 + volume surge
            elif (price < s1_1d_6h[i] and price < ema_50_1w_6h[i] and 
                  vol > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long exit: price crosses below R1 OR below weekly EMA50
            if price < r1_1d_6h[i] or price < ema_50_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above S1 OR above weekly EMA50
            if price > s1_1d_6h[i] or price > ema_50_1w_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R1S1_WeeklyEMA50_Volume"
timeframe = "6h"
leverage = 1.0