#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation
# The Alligator (Jaw/Teeth/Lips) identifies trend absence when lines are intertwined.
# Trade only when Lips > Teeth > Jaw (bull) or Lips < Teeth < Jaw (bear) with 1d EMA50 alignment.
# Requires 1d volume spike to confirm institutional participation.
# Designed for low-frequency, high-conviction trades to minimize fee drag on 6h timeframe.

name = "6h_Alligator_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filters
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d volume spike (2x 20-period average)
    vol_ma_1d = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (vol_ma_1d * 2.0)
    vol_spike_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d)
    
    # Williams Alligator on 6h (13,8,5 SMAs shifted)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars  
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw_raw = smma(close, 13)
    teeth_raw = smma(close, 8)
    lips_raw = smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    # Fill NaNs from rolling
    jaw[:8] = np.nan
    teeth[:5] = np.nan
    lips[:3] = np.nan
    
    # Align Alligator lines to 6s timeframe (already aligned via smma on close)
    jaw_aligned = jaw
    teeth_aligned = teeth
    lips_aligned = lips
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(13, 8, 5) + 8  # Ensure all Alligator data available
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter bullish alignment: Lips > Teeth > Jaw AND price above 1d EMA50 AND volume spike
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter bearish alignment: Lips < Teeth < Jaw AND price below 1d EMA50 AND volume spike
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < ema50_1d_aligned[i] and vol_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator lines intertwine (Lips < Teeth or Teeth < Jaw) OR trend fails
            if (lips_aligned[i] < teeth_aligned[i] or teeth_aligned[i] < jaw_aligned[i] or
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator lines intertwine (Lips > Teeth or Teeth > Jaw) OR trend fails
            if (lips_aligned[i] > teeth_aligned[i] or teeth_aligned[i] > jaw_aligned[i] or
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals