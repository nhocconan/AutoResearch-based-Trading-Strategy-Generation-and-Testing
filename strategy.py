#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d Trend Filter + Volume Spike
# Williams Alligator uses smoothed moving averages (Jaw/Teeth/Lips) to identify trends.
# Trend is bullish when Lips > Teeth > Jaw, bearish when Lips < Teeth < Jaw.
# 1d EMA50 filter ensures we trade with higher timeframe momentum.
# Volume spike confirms institutional participation.
# Target: 20-50 total trades over 4 years (5-12/year) to avoid fee drag.
name = "4h_WilliamsAlligator_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: 3 SMMA (Smoothed Moving Average)
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead
    # Lips: 5-period SMMA, 3 bars ahead
    # SMMA formula: SMMA(t) = (SMMA(t-1) * (n-1) + price(t)) / n
    
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # 20-period volume average for spike detection
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA50 to 4h
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema50_1d_4h[i]) or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume condition: current volume > 2.0 x 20-period average
        vol_spike = volume[i] > vol_avg[i] * 2.0
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + above 1d EMA50 + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1d_4h[i] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + below 1d EMA50 + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_1d_4h[i] and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Alligator alignment breaks OR price below 1d EMA50
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Alligator alignment breaks OR price above 1d EMA50
            if not (lips[i] < teeth[i] and teeth[i] < jaw[i]) or close[i] > ema50_1d_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals