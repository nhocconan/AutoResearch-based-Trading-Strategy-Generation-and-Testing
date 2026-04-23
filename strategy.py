#!/usr/bin/env python3
"""
Hypothesis: 6h Williams Alligator + 1d Elder Ray (Bull/Bear Power) + volume confirmation.
Long when Alligator is bullish (JAW > TEETH > LIPS) AND 1d Bull Power > 0 AND volume > 1.5x 20-period average.
Short when Alligator is bearish (LIPS < TEETH < JAW) AND 1d Bear Power < 0 AND volume > 1.5x 20-period average.
Exit when Alligator becomes neutral (jaws cross teeth or lips) OR volume drops below average.
Uses discrete position sizing (0.25) to minimize fee churn. Targets 12-37 trades/year per symbol.
Williams Alligator identifies trend structure via smoothed medians. Elder Ray measures bull/bear power relative to 13-period EMA.
Combines trend confirmation (Alligator) with momentum (Elder Ray) and volume filter for high-probability entries in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 13-period EMA for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Williams Alligator on 6h timeframe (smoothed medians)
    # Jaw: 13-period SMMA shifted 8 bars
    # Teeth: 8-period SMMA shifted 5 bars
    # Lips: 5-period SMMA shifted 3 bars
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    # Calculate Alligator components
    median_price = (high + low) / 2.0
    jaw = smma(median_price, 13)  # 13-period SMMA
    teeth = smma(median_price, 8)   # 8-period SMMA
    lips = smma(median_price, 5)    # 5-period SMMA
    
    # Shift as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Align HTF Elder Ray to 6h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 13+8, 20)  # Ensure warmup for SMMA and EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Alligator bullish AND Bull Power positive AND volume spike
            if (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i] and  # Jaw > Teeth > Lips
                bull_power_1d_aligned[i] > 0 and
                volume[i] > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Alligator bearish AND Bear Power negative AND volume spike
            elif (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i] and  # Lips < Teeth < Jaw
                  bear_power_1d_aligned[i] < 0 and
                  volume[i] > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            # Primary exit: Alligator loses alignment (jaws cross teeth or lips)
            if position == 1 and not (jaw_shifted[i] > teeth_shifted[i] > lips_shifted[i]):
                exit_signal = True
            elif position == -1 and not (lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]):
                exit_signal = True
            
            # Secondary exit: volume drops below average (loss of momentum)
            elif volume[i] < vol_ma_val:
                exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsAlligator_1dElderRay_VolumeConfirm"
timeframe = "6h"
leverage = 1.0