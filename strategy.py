#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator + 1d EMA50 trend filter + volume spike (>1.8x 20-bar avg) for confirmation.
# Uses 1d EMA50 for HTF trend alignment, 12h Williams Alligator (Jaw/Teeth/Lips) for trend strength and direction,
# and volume confirmation to avoid false signals. Designed for low trade frequency (target 50-150 total over 4 years)
# to minimize fee drag while capturing strong trends in both bull and bear markets by following the 1d trend.

name = "12h_WilliamsAlligator_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h (primary TF)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    def smma(src, length):
        # Smoothed Moving Average: first value is SMA, then recursive
        result = np.full_like(src, np.nan, dtype=np.float64)
        if len(src) < length:
            return result
        # First value: simple moving average
        result[length-1] = np.mean(src[:length])
        # Subsequent values: SMMA = (PREV_SMMA * (LENGTH-1) + CURRENT_CLOSE) / LENGTH
        for i in range(length, len(src)):
            result[i] = (result[i-1] * (length-1) + src[i]) / length
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Calculate average volume for confirmation (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # start after lookback
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaw (Alligator bullish alignment), close > 1d EMA50, volume spike (>1.8x avg)
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume[i] > 1.8 * avg_volume[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaw (Alligator bearish alignment), close < 1d EMA50, volume spike (>1.8x avg)
            elif (lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume[i] > 1.8 * avg_volume[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Close position if Alligator turns bearish (Lips < Teeth) or close < 1d EMA50
            if lips[i] < teeth[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Close position if Alligator turns bullish (Lips > Teeth) or close > 1d EMA50
            if lips[i] > teeth[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals