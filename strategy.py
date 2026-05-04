#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation
# Uses 1d EMA50 for higher timeframe trend alignment to reduce whipsaw
# Williams Alligator (Jaw=13, Teeth=8, Lips=5) provides smoothed trend signals
# Volume confirmation (>1.8x 20 EMA) filters low-participation false signals
# Discrete sizing 0.25 limits risk and reduces fee churn
# Target: 50-150 total trades over 4 years = 12-37/year for 12h.
# Works in both bull and bear: trend filter adapts to higher timeframe direction.

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend direction
    close_1d = pd.Series(df_1d['close'])
    ema50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 12h timeframe (completed 1d bar only)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator on 12h timeframe
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=float)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (PERIOD-1) + CLOSE) / PERIOD
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Volume confirmation: 20-period EMA of volume on 12h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema50_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Lips > Teeth > Jaw (bullish alignment) + uptrend + volume spike
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Lips < Teeth < Jaw (bearish alignment) + downtrend + volume spike
            elif lips[i] < teeth[i] and teeth[i] < jaw[i] and close[i] < ema50_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines converge OR trend changes OR weak volume
            if (lips[i] <= teeth[i] or  # Alligator lines converging/crossing
                close[i] < ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines converge OR trend changes OR weak volume
            if (lips[i] >= teeth[i] or  # Alligator lines converging/crossing
                close[i] > ema50_aligned[i] or 
                volume[i] < vol_ema_20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals