#!/usr/bin/env python3
"""
4h_Williams_Alligator_Trend_With_Volume_Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction. When Lips cross above Teeth with volume confirmation and price above Jaw (bullish alignment), go long. Reverse for short. Uses 25% position size to balance risk/return and limit trade frequency (~20-40/year) to minimize fee drag in 4-hour bars. Williams Alligator is less common than EMA-based systems, offering potential edge in both bull and bear markets via clear trend alignment signals.
"""

name = "4h_Williams_Alligator_Trend_With_Volume_Spike"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter (once before loop)
    df_12h = get_htf_data(prices, '12h')
    
    # Williams Alligator: Jaw (13-period SMMA, 8-bar shift), Teeth (8-period SMMA, 5-bar shift), Lips (5-period SMMA, 3-bar shift)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
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
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift as per Williams Alligator specification
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # 12h trend filter: EMA(50) on close
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after Alligator warmup
        if position == 0:
            # LONG: Lips cross above Teeth, price above Jaw (bullish alignment), volume confirmation
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1] and 
                close[i] > jaw[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Lips cross below Teeth, price below Jaw (bearish alignment), volume confirmation
            elif (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1] and 
                  close[i] < jaw[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Lips cross below Teeth OR price below Jaw OR volume drops
            if (lips[i] < teeth[i] and lips[i-1] >= teeth[i-1]) or \
               close[i] < jaw[i] or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Lips cross above Teeth OR price above Jaw OR volume drops
            if (lips[i] > teeth[i] and lips[i-1] <= teeth[i-1]) or \
               close[i] > jaw[i] or \
               not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals