#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-day EMA(50) trend filter
# Uses Alligator (Jaw/Teeth/Lips) to detect trends and EMA(50) from 1d to filter direction.
# Designed to work in both bull and bear markets via trend filter and avoids whipsaw with Alligator's alignment requirement.
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag.

name = "6h_williams_alligator_1d_ema50_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Williams Alligator on 6h: Jaw(13,8), Teeth(8,5), Lips(5,3)
    # Jaw: 13-period SMMA, 8-period shift
    # Teeth: 8-period SMMA, 5-period shift  
    # Lips: 5-period SMMA, 3-period shift
    def smma(series, period):
        # Smoothed Moving Average (SMMA) = EMA with alpha = 1/period
        return pd.Series(series).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    # Fill NaN from roll with first valid value
    for arr in [jaw, teeth, lips]:
        if len(arr) > 0:
            arr[0:8 if arr is jaw else 5 if arr is teeth else 3] = arr[8 if arr is jaw else 5 if arr is teeth else 3]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: Alligator lines not aligned (Lips < Teeth or Teeth < Jaw) OR price below 1d EMA(50)
            if lips[i] < teeth[i] or teeth[i] < jaw[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Alligator lines not aligned (Lips > Teeth or Teeth > Jaw) OR price above 1d EMA(50)
            if lips[i] > teeth[i] or teeth[i] > jaw[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Alligator aligned: Lips > Teeth > Jaw (uptrend) or Lips < Teeth < Jaw (downtrend)
            alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
            alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
            
            # Long: Alligator aligned up AND price above 1d EMA(50)
            if alligator_long and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down AND price below 1d EMA(50)
            elif alligator_short and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
    
    return signals