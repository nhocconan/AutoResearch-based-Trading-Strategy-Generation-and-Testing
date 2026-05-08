#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator with 1d trend filter and volume confirmation
# Uses Alligator (Jaw/Teeth/Lips) to identify trends, 1d EMA50 for trend filter,
# and volume spike for confirmation. Long when Lips > Teeth > Jaw and price above Jaw,
# short when Lips < Teeth < Jaw and price below Jaw. Avoids chop by requiring
# volume > 1.5x average. Designed for fewer trades (20-50/year) to avoid fee drag.
# Works in bull (follow Alligator alignment) and bear (reverse alignment).

name = "4h_WilliamsAlligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Williams Alligator (SMMA with specific periods)
    def smma(arr, period):
        # Smoothed Moving Average: similar to EMA but with alpha = 1/period
        res = np.full_like(arr, np.nan)
        if len(arr) < period:
            return res
        res[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            res[i] = (res[i-1] * (period-1) + arr[i]) / period
        return res
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # 1d EMA(50) for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1d_up = ema_50_1d[1:] > ema_50_1d[:-1]
    trend_1d_up = np.concatenate([[False], trend_1d_up])
    
    # Align 1d trend to 4h
    trend_1d_up_aligned = align_htf_to_ltf(prices, df_1d, trend_1d_up.astype(float))
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(trend_1d_up_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) AND price above Jaw AND volume spike AND 1d uptrend
            if (lips[i] > teeth[i] > jaw[i] and 
                close[i] > jaw[i] and 
                volume_spike[i] and 
                trend_1d_up_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) AND price below Jaw AND volume spike AND 1d downtrend
            elif (lips[i] < teeth[i] < jaw[i] and 
                  close[i] < jaw[i] and 
                  volume_spike[i] and 
                  not trend_1d_up_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: alignment breaks or price crosses below Jaw
            if not (lips[i] > teeth[i] > jaw[i]) or close[i] <= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: alignment breaks or price crosses above Jaw
            if not (lips[i] < teeth[i] < jaw[i]) or close[i] >= jaw[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals