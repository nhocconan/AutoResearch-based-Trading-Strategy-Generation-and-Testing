#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d 15-period EMA trend + 1w pivot rejection + volume confirmation.
# Long when price > 1d EMA15 AND price rejects below 1w pivot (bounce) AND volume > 1.5x average.
# Short when price < 1d EMA15 AND price rejects above 1w pivot (rejection) AND volume > 1.5x average.
# Exit when price crosses back through the 1d EMA15.
# Uses EMA for trend, pivot for mean reversion zones, volume for confirmation.
# Designed to work in bull (buy dips to EMA) and bear (sell rallies to EMA) markets.
# Target: 15-35 trades/year to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA15 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 15-period EMA
    ema_15 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 15:
        ema_15[14] = np.mean(close_1d[:15])  # Simple average for first value
        alpha = 2.0 / (15 + 1)
        for i in range(15, len(close_1d)):
            ema_15[i] = alpha * close_1d[i] + (1 - alpha) * ema_15[i-1]
    
    # Load 1w data for pivot point calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot point: (H + L + C) / 3
    pivot_1w = np.full_like(high_1w, np.nan)
    for i in range(len(high_1w)):
        pivot_1w[i] = (high_1w[i] + low_1w[i] + close_1w[i]) / 3.0
    
    # Load 6h data for volume confirmation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    volume_6h = df_6h['volume'].values
    
    # Calculate 20-period average volume
    vol_ma_20 = np.full_like(volume_6h, np.nan)
    for i in range(19, len(volume_6h)):
        vol_ma_20[i] = np.mean(volume_6h[i-19:i+1])
    
    # Align indicators to 6h timeframe
    ema_15_aligned = align_htf_to_ltf(prices, df_1d, ema_15)
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_6h, vol_ma_20)
    volume_6h_aligned = align_htf_to_ltf(prices, df_6h, volume_6h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(15, 19)  # Need EMA15 and volume MA20
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_15_aligned[i]) or 
            np.isnan(pivot_1w_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(volume_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        volume_ratio = volume_6h_aligned[i] / vol_ma_20_aligned[i] if vol_ma_20_aligned[i] > 0 else 0
        
        if position == 0:
            # Look for entries: EMA trend + pivot rejection + volume confirmation
            # Long: price above EMA15 AND price rejects from below pivot (bounce) AND volume confirmation
            if (close[i] > ema_15_aligned[i] and 
                low[i] <= pivot_1w_aligned[i] * 1.002 and  # Allow small tolerance for wick
                close[i] > pivot_1w_aligned[i] and  # Close above pivot confirms bounce
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price below EMA15 AND price rejects from above pivot (rejection) AND volume confirmation
            elif (close[i] < ema_15_aligned[i] and 
                  high[i] >= pivot_1w_aligned[i] * 0.998 and  # Allow small tolerance for wick
                  close[i] < pivot_1w_aligned[i] and  # Close below pivot confirms rejection
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below EMA15
            if close[i] < ema_15_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses back above EMA15
            if close[i] > ema_15_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_EMA15_1wPivot_Rejection_Volume_v1"
timeframe = "6h"
leverage = 1.0