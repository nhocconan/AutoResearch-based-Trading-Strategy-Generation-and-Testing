#!/usr/bin/env python3
# 1h_Camarilla_R1_S1_Breakout_4hTrend_Volume
# Hypothesis: Use 4h Camarilla pivot breakouts (R1/S1) as entry signals, filtered by 4h EMA50 trend and volume spike.
# Enter long when price breaks above R1 with volume > 1.5x 20-period average and price > EMA50.
# Enter short when price breaks below S1 with volume > 1.5x 20-period average and price < EMA50.
# Exit on opposite breakout or volume drop. Uses 1h only for entry timing, 4h for trend/volume.
# Designed for low frequency (15-35 trades/year) to avoid fee drag. Camarilla pivots work in ranging markets,
# EMA filter avoids false breakouts in weak trends, volume confirms institutional participation.

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """
    Calculate Camarilla pivot levels: R1, R2, R3, S1, S2, S3.
    Based on previous period's high, low, close.
    Returns R1, S1 arrays.
    """
    n = len(close)
    R1 = np.full(n, np.nan)
    S1 = np.full(n, np.nan)
    
    # Camarilla uses previous period's data
    for i in range(1, n):
        high_prev = high[i-1]
        low_prev = low[i-1]
        close_prev = close[i-1]
        
        # Calculate pivot point
        pivot = (high_prev + low_prev + close_prev) / 3
        
        # Calculate ranges
        range_val = high_prev - low_prev
        
        # Camarilla levels
        R1[i] = close_prev + (range_val * 1.1 / 12)
        S1[i] = close_prev - (range_val * 1.1 / 12)
    
    return R1, S1

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and volume filters
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate 4h Camarilla levels (R1, S1)
    R1_4h, S1_4h = calculate_camarilla(high_4h, low_4h, close_4h)
    
    # Calculate 4h EMA50 for trend filter
    close_4h_series = pd.Series(close_4h)
    ema50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume average (20-period) for volume spike filter
    volume_4h_series = pd.Series(volume_4h)
    vol_avg_20_4h = volume_4h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align 4h indicators to 1h timeframe
    R1_4h_aligned = align_htf_to_ltf(prices, df_4h, R1_4h)
    S1_4h_aligned = align_htf_to_ltf(prices, df_4h, S1_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    vol_avg_20_4h_aligned = align_htf_to_ltf(prices, df_4h, vol_avg_20_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure EMA50 and volume average are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(R1_4h_aligned[i]) or np.isnan(S1_4h_aligned[i]) or 
            np.isnan(ema50_4h_aligned[i]) or np.isnan(vol_avg_20_4h_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current 4h volume > 1.5x 20-period average
        volume_spike = volume_4h[i//4] > 1.5 * vol_avg_20_4h_aligned[i] if i//4 < len(volume_4h) else False
        
        if position == 0:
            # LONG: Price breaks above R1 AND volume spike AND price > EMA50
            if high[i] > R1_4h_aligned[i] and volume_spike and close[i] > ema50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below S1 AND volume spike AND price < EMA50
            elif low[i] < S1_4h_aligned[i] and volume_spike and close[i] < ema50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # EXIT LONG: Price breaks below S1 OR volume drops
            if low[i] < S1_4h_aligned[i] or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above R1 OR volume drops
            if high[i] > R1_4h_aligned[i] or not volume_spike:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals