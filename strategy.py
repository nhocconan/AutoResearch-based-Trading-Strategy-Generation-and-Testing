#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R4/S4 breakout with 1w pivot direction filter and volume confirmation
# Long when price breaks above Camarilla R4 with 1w bullish pivot structure and volume > 2.0x 20-period volume EMA
# Short when price breaks below Camarilla S4 with 1w bearish pivot structure and volume > 2.0x 20-period volume EMA
# Uses 1w HTF pivot structure for major trend filter to reduce whipsaw vs shorter HTF, targeting 12-37 trades/year on 6h.
# Volume spike filter (2.0x) is strict to avoid overtrading. Camarilla R4/S4 are strong breakout levels.
# Works in bull markets via longs in bullish 1w pivot regime and bear markets via shorts in bearish 1w pivot regime.

name = "6h_Camarilla_R4S4_1wPivot_Direction_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for HTF pivot structure - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w pivot points (standard floor pivot)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    r2_1w = pivot_1w + (high_1w - low_1w)
    s2_1w = pivot_1w - (high_1w - low_1w)
    r3_1w = high_1w + 2 * (pivot_1w - low_1w)
    s3_1w = low_1w - 2 * (high_1w - pivot_1w)
    
    # Determine 1w pivot direction: bullish if close > pivot, bearish if close < pivot
    pivot_bullish_1w = close_1w > pivot_1w
    pivot_bearish_1w = close_1w < pivot_1w
    
    # Align 1w pivot direction to 6h timeframe
    pivot_bullish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bullish_1w.astype(float))
    pivot_bearish_aligned = align_htf_to_ltf(prices, df_1w, pivot_bearish_1w.astype(float))
    
    # Get 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R4 and S4 from previous 1d bar
    camarilla_range = (high_1d - low_1d) * 1.1
    camarilla_r4 = close_1d + camarilla_range / 2  # R4 = close + 1.1*(high-low)*1.1/2
    camarilla_s4 = close_1d - camarilla_range / 2  # S4 = close - 1.1*(high-low)*1.1/2
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Calculate volume spike filter (20-period volume EMA)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ema_20 * 2.0)  # Volume at least 2.0x average for confirmation
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(pivot_bullish_aligned[i]) or np.isnan(pivot_bearish_aligned[i]) or 
            np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R4 AND 1w bullish pivot AND volume spike
            if (close[i] > camarilla_r4_aligned[i] and 
                pivot_bullish_aligned[i] > 0.5 and  # 1w bullish pivot
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Camarilla S4 AND 1w bearish pivot AND volume spike
            elif (close[i] < camarilla_s4_aligned[i] and 
                  pivot_bearish_aligned[i] > 0.5 and  # 1w bearish pivot
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price closes below Camarilla S4 OR 1w pivot turns bearish
            if (close[i] < camarilla_s4_aligned[i] or 
                pivot_bearish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price closes above Camarilla R4 OR 1w pivot turns bullish
            if (close[i] > camarilla_r4_aligned[i] or 
                pivot_bullish_aligned[i] > 0.5):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals