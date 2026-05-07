#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator combined with 1d EMA50 trend filter.
# Long when: Alligator is bullish (Green line > Red line > Blue line) AND price > 1d EMA50.
# Short when: Alligator is bearish (Green line < Red line < Blue line) AND price < 1d EMA50.
# Exit when Alligator lines re-cross or price crosses 1d EMA50 in opposite direction.
# Williams Alligator uses smoothed medians (Jaw=TEETH=13, TEETH=8, LIPS=5) to filter noise.
# The 1d EMA50 ensures we trade with the daily trend, reducing whipsaws in sideways markets.
# This combination captures trending moves while avoiding false signals in chop.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
# Works in both bull and bear markets by following the 1d EMA50 trend.

name = "6h_WilliamsAlligator_1dEMA50_Trend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Daily data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Williams Alligator: 3 smoothed medians (using close as proxy for median)
    # Jaw (Blue): 13-period SMMA, 8 bars ahead
    # Teeth (Red): 8-period SMMA, 5 bars ahead
    # Lips (Green): 5-period SMMA, 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    def smma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        result = np.full_like(series, np.nan)
        sma = np.mean(series[:period])
        result[period-1] = sma
        for i in range(period, len(series)):
            result[i] = (result[i-1] * (period-1) + series[i]) / period
        return result
    
    # Calculate Alligator lines from 6h data
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift to avoid look-ahead (Alligator uses future values in its calculation)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Set NaN for shifted values
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Alligator conditions
    alligator_bullish = (lips > teeth) & (teeth > jaw)  # Green > Red > Blue
    alligator_bearish = (lips < teeth) & (teeth < jaw)  # Green < Red < Blue
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13)  # Sufficient warmup for EMA50 and Alligator
    
    for i in range(start_idx, n):
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(lips[i]) or np.isnan(teeth[i]) or 
            np.isnan(jaw[i]) or np.isnan(alligator_bullish[i]) or np.isnan(alligator_bearish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Alligator bullish AND price > 1d EMA50
            long_cond = alligator_bullish[i] and (close[i] > ema50_1d_aligned[i])
            # Short conditions: Alligator bearish AND price < 1d EMA50
            short_cond = alligator_bearish[i] and (close[i] < ema50_1d_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator turns bearish OR price crosses below 1d EMA50
            if (not alligator_bullish[i]) or (close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator turns bullish OR price crosses above 1d EMA50
            if (not alligator_bearish[i]) or (close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals