#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 12h trend filter and volume spike.
Works in bull/bear markets because: 1) Alligator identifies trend vs range (jaws/teeth/lips), 
2) 12h EMA filter ensures alignment with higher timeframe trend, 
3) Volume spike confirms breakout strength, reducing false signals. Target 20-35 trades/year.
"""
name = "4h_WilliamsAlligator_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h DATA FOR TREND FILTER ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # === WILLIAMS ALLIGATOR (13,8,5 SMMA) ===
    # SMMA (Smoothed Moving Average) - similar to Wilder's smoothing
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Align Alligator lines to 4h
    jaws_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), jaws)
    teeth_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), teeth)
    lips_aligned = align_htf_to_ltf(prices, pd.DataFrame({'close': close}), lips)
    
    # === VOLUME CONFIRMATION (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 13)  # 50 for 12h EMA, 13 for Alligator jaws
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(jaws_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Lips > Teeth > Jaws (bullish alignment) AND price above 12h EMA50 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] and 
                teeth_aligned[i] > jaws_aligned[i] and
                close[i] > ema50_12h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips < Teeth < Jaws (bearish alignment) AND price below 12h EMA50 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] and 
                  teeth_aligned[i] < jaws_aligned[i] and
                  close[i] < ema50_12h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: Alligator alignment breaks (lips < teeth) OR price below 12h EMA50
            if (lips_aligned[i] < teeth_aligned[i]) or (close[i] < ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Alligator alignment breaks (lips > teeth) OR price above 12h EMA50
            if (lips_aligned[i] > teeth_aligned[i]) or (close[i] > ema50_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals