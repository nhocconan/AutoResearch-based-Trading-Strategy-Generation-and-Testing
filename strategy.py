#!/usr/bin/env python3
"""
4h_camarilla_pivot_12h_volume_v1
Hypothesis: On 4-hour timeframe, use daily Camarilla pivot levels with 12-hour trend filter and volume confirmation.
Long when price touches S1/S2 support with 12h EMA(20) trending up and volume > 1.5x 20-period average.
Short when price touches R1/R2 resistance with 12h EMA(20) trending down and volume > 1.5x 20-period average.
Exit when price moves back toward the mean (Pivot point) or reverses at opposite levels.
Designed for 20-40 trades/year to minimize fee drag while capturing mean-reversion bounces in ranging markets and trend continuations in trending markets.
Works in both bull/bear markets as Camarilla levels adapt to volatility and 12h trend filter avoids counter-trend trades.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_12h_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels: R4 = C + ((H-L) * 1.5000), R3 = C + ((H-L) * 1.2500), etc.
    r4 = pivot + (range_1d * 1.5000)
    r3 = pivot + (range_1d * 1.2500)
    r2 = pivot + (range_1d * 1.1666)
    r1 = pivot + (range_1d * 1.0833)
    s1 = pivot - (range_1d * 1.0833)
    s2 = pivot - (range_1d * 1.1666)
    s3 = pivot - (range_1d * 1.2500)
    s4 = pivot - (range_1d * 1.5000)
    
    # Align pivots to 4h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    close_12h = df_12h['close'].values
    ema_20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Determine 12h trend direction (using EMA slope)
    trend_up = np.zeros(len(ema_20_12h_aligned), dtype=bool)
    trend_down = np.zeros(len(ema_20_12h_aligned), dtype=bool)
    for i in range(1, len(ema_20_12h_aligned)):
        if not np.isnan(ema_20_12h_aligned[i]) and not np.isnan(ema_20_12h_aligned[i-1]):
            trend_up[i] = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            trend_down[i] = ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]
    
    # Volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_20_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
            
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price moves back to pivot or breaks below S1
            if close[i] <= pivot_aligned[i] or close[i] < s1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves back to pivot or breaks above R1
            if close[i] >= pivot_aligned[i] or close[i] > r1_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter with volume confirmation and 12h trend alignment
            if vol_ok:
                # Long: price touches S1/S2 with 12h uptrend
                if ((abs(close[i] - s1_aligned[i]) < 0.001 * close[i] or 
                     abs(close[i] - s2_aligned[i]) < 0.001 * close[i]) and 
                    trend_up[i]):
                    position = 1
                    signals[i] = 0.25
                # Short: price touches R1/R2 with 12h downtrend
                elif ((abs(close[i] - r1_aligned[i]) < 0.001 * close[i] or 
                       abs(close[i] - r2_aligned[i]) < 0.001 * close[i]) and 
                      trend_down[i]):
                    position = -1
                    signals[i] = -0.25
    
    return signals