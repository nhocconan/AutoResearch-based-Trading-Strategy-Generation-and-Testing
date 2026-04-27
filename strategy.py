# State the hypothesis and strategy
# Hypothesis: A 12-hour strategy using the Williams Alligator (SMMA-based) to identify trends and trigger entries on price breaks of the Alligator's teeth (middle line) with volume confirmation. The Alligator acts as a trend filter and dynamic support/resistance. Works in bull markets by catching trends and in bear markets by avoiding false signals via the Alligator's sleep/awake phases. Targets 15-25 trades/year to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - used in Williams Alligator"""
    result = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return result
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Williams Alligator (13,8,5 periods)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Williams Alligator lines: Jaw (13), Teeth (8), Lips (5) - all SMMA
    close_1d = df_1d['close'].values
    jaw = smma(close_1d, 13)   # Blue line
    teeth = smma(close_1d, 8)  # Red line
    lips = smma(close_1d, 5)   # Green line
    
    # Align to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for Alligator (13 period) and volume MA
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            signals[i] = 0.0
            continue
        
        jaw_val = jaw_aligned[i]
        teeth_val = teeth_aligned[i]
        lips_val = lips_aligned[i]
        vol_spike_val = vol_spike[i]
        
        # Alligator sleep condition: all lines intertwined (no trend)
        # When lips, teeth, and jaw are close, market is sleeping
        alligator_sleep = (abs(lips_val - teeth_val) < 0.001 * jaws_val and 
                          abs(teeth_val - jaw_val) < 0.001 * jaws_val)
        
        if position == 0:
            # Only enter if Alligator is awake (not sleeping) and volume spike
            if not alligator_sleep and vol_spike_val:
                # Long: price above teeth and lips above jaws (bullish alignment)
                if close[i] > teeth_val and lips_val > jaw_val:
                    signals[i] = size
                    position = 1
                # Short: price below teeth and lips below jaws (bearish alignment)
                elif close[i] < teeth_val and lips_val < jaw_val:
                    signals[i] = -size
                    position = -1
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below teeth or Alligator starts sleeping
            if close[i] < teeth_val or alligator_sleep:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above teeth or Alligator starts sleeping
            if close[i] > teeth_val or alligator_sleep:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_WilliamsAlligator_TeethBreak_Volume"
timeframe = "12h"
leverage = 1.0