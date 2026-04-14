#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    """
    Hypothesis: In 6h timeframe, price respects 1-day pivot levels (S3/R3) as dynamic support/resistance.
    Breakouts above R3 with volume confirmation indicate bullish momentum.
    Breakdowns below S3 with volume confirmation indicate bearish momentum.
    Uses 1-day ATR as volatility filter to avoid low-volatility whipsaws.
    Works in both bull and bear markets by following breakouts from key daily levels.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1-day ATR (14-period) for volatility filter
    high_low = high_1d - low_1d
    high_close = np.abs(high_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    low_close = np.abs(low_1d - np.concatenate([[close_1d[0]], close_1d[:-1]]))
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-day pivot points (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align S3 and R3 to 6h timeframe (these are our key levels)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    
    # Align ATR to 6h timeframe for volatility filter
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6-hour volume moving average (20-period)
    volume_ma = np.full(n, np.nan)
    if n >= 20:
        for i in range(19, n):
            volume_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(s3_6h[i]) or np.isnan(r3_6h[i]) or 
            np.isnan(atr_6h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low-volatility periods (ATR < 0.4% of price)
        if atr_6h[i] / close[i] < 0.004:
            signals[i] = 0.0
            continue
        
        # Volume filter: require volume > 60% of 20-period MA
        if volume[i] < 0.6 * volume_ma[i]:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R3 with volume confirmation
            if close[i] > r3_6h[i]:
                position = 1
                signals[i] = position_size
            # Short: Price breaks below S3 with volume confirmation
            elif close[i] < s3_6h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price falls back below S3 (failed breakout) or volatility drops
            if close[i] < s3_6h[i] or atr_6h[i] / close[i] < 0.003:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price rises back above R3 (failed breakdown) or volatility drops
            if close[i] > r3_6h[i] or atr_6h[i] / close[i] < 0.003:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1d_Pivot_S3R3_Breakout_Volume_Filter"
timeframe = "6h"
leverage = 1.0