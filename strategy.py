# 1d_1w_Williams_Alligator_Trend_Follow_v1
# Hypothesis: Use Williams Alligator (3 SMAs: Jaw=13, Teeth=8, Lips=5) on weekly timeframe to identify strong trends.
# In trending markets (JAW > TEETH > LIPS for long, JAW < TEETH < LIPS for short), enter on pullbacks to the TEETH (8 SMA) on daily timeframe.
# Uses volume confirmation (20-period MA) and avoids choppy markets via Alligator alignment.
# Designed for low trade frequency (10-25 trades/year) to minimize fee drag in both bull and bear markets.

name = "1d_1w_Williams_Alligator_Trend_Follow_v1"
timeframe = "1d"
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
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate Williams Alligator on weekly data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Jaw (13-period SMMA), Teeth (8-period SMMA), Lips (5-period SMMA)
    def smma(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        # First value: simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: smoothed moving average
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1w, 13)  # Blue line
    teeth = smma(close_1w, 8)  # Red line
    lips = smma(close_1w, 5)   # Green line
    
    # Align Alligator lines to daily timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1w, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1w, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1w, lips)
    
    # Daily TEETH (8 SMA) for entry pullbacks
    teeth_daily = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(teeth_daily[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long condition: Alligator aligned bullish (JAW > TEETH > LIPS) + pullback to TEETH + volume
            if (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i] and
                low[i] <= teeth_daily[i] * 1.01 and high[i] >= teeth_daily[i] * 0.99 and
                volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short condition: Alligator aligned bearish (JAW < TEETH < LIPS) + pullback to TEETH + volume
            elif (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i] and
                  low[i] <= teeth_daily[i] * 1.01 and high[i] >= teeth_daily[i] * 0.99 and
                  volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator loses bullish alignment OR price closes below LIPS
            if not (jaw_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > lips_aligned[i]) or \
               close[i] < lips_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator loses bearish alignment OR price closes above JAW
            if not (jaw_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < lips_aligned[i]) or \
               close[i] > jaw_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals