#!/usr/bin/env python3
"""
Hypothesis: 12h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator: Jaw (13-period SMMA shifted 8), Teeth (8-period SMMA shifted 5), Lips (5-period SMMA shifted 3)
- Long: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume > 1.5x 20-period avg + price > 1d EMA50
- Short: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume > 1.5x 20-period avg + price < 1d EMA50
- Exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw crossover) or price closes opposite the Alligator mouth
- Uses 1d EMA50 for higher timeframe trend filter to avoid counter-trend trades
- Williams Alligator catches trends early and stays in them, reducing whipsaw vs MA crossovers
- Volume confirmation ensures breakouts have participation
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(values, period):
    """Smoothed Moving Average (SMMA) / Wilder's MA"""
    if len(values) < period:
        return np.full_like(values, np.nan, dtype=float)
    result = np.full_like(values, np.nan, dtype=float)
    # First value is SMA
    result[period-1] = np.mean(values[:period])
    # Subsequent values: SMMA = (Prev SMMA * (period-1) + Current Close) / period
    for i in range(period, len(values)):
        result[i] = (result[i-1] * (period-1) + values[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Alligator lines: SMMA with different periods and shifts
    jaw = smma(close_12h, 13)  # Jaw: 13-period SMMA
    teeth = smma(close_12h, 8)  # Teeth: 8-period SMMA
    lips = smma(close_12h, 5)   # Lips: 5-period SMMA
    
    # Apply shifts (Jaw shifted 8 bars, Teeth 5 bars, Lips 3 bars)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First shifted values will be invalid due to roll, set to NaN
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Align Alligator lines to 12h timeframe (wait for 12h bar close)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips_shifted)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 13+8, 8+5, 5+3)  # Volume MA(20), Alligator max shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume spike + price > 1d EMA50
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                close[i] > lips_aligned[i] and 
                volume_spike and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume spike + price < 1d EMA50
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  close[i] < lips_aligned[i] and 
                  volume_spike and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw) or price closes below Jaw
            if (lips_aligned[i] <= teeth_aligned[i] or 
                teeth_aligned[i] <= jaw_aligned[i] or 
                close[i] < jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Alligator lines cross (Lips-Teeth or Teeth-Jaw) or price closes above Jaw
            if (lips_aligned[i] >= teeth_aligned[i] or 
                teeth_aligned[i] >= jaw_aligned[i] or 
                close[i] > jaw_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0