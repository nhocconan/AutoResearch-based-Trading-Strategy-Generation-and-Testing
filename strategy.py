#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Elder Ray + Volume Confirmation
# - Uses 1d Williams Alligator (Jaw=TEETH=13, Teeth=8, Lips=5) smoothed with SMMA
# - 1d Elder Ray: Bull Power = High - EMA13, Bear Power = Lows - EMA13
# - Enter Long when: Alligator aligned bullish (Lips > Teeth > Jaw) AND Bull Power > 0 AND Volume > 1.5x 20-period average
# - Enter Short when: Alligator aligned bearish (Lips < Teeth < Jaw) AND Bear Power < 0 AND Volume > 1.5x 20-period average
# - Exit when Alligator alignment reverses or volume drops below average
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Targets 50-150 total trades over 4 years (12-37/year) to avoid fee drag
# - Works in both bull and bear markets: Alligator identifies trend, Elder Ray measures power, Volume confirms conviction

name = "6h_1d_alligator_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Williams Alligator: SMMA (Smoothed Moving Average)
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Alligator lines: Jaw (13,8), Teeth (8,5), Lips (5,3)
    jaw = smma(close_1d, 13)
    teeth = smma(close_1d, 8)
    lips = smma(close_1d, 5)
    
    # Shift Alligator lines by 5, 3, 0 periods respectively (as per Williams)
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    # Lips not shifted
    
    # Align Alligator to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: > 1.5x 20-period average
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume_1d > (1.5 * avg_volume_20_1d)
    vol_confirm_aligned = align_htf_to_ltf(prices, df_1d, vol_confirm)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(lips_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(jaw_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_confirm_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for entry
            # Bullish condition: Lips > Teeth > Jaw (Alligator aligned up) AND Bull Power > 0 AND Volume confirmed
            if (lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i] and 
                bull_power_aligned[i] > 0 and 
                vol_confirm_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Bearish condition: Lips < Teeth < Jaw (Alligator aligned down) AND Bear Power < 0 AND Volume confirmed
            elif (lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i] and 
                  bear_power_aligned[i] < 0 and 
                  vol_confirm_aligned[i]):
                position = -1
                signals[i] = -0.25
        
        elif position == 1:  # Long position - exit when Alligator alignment breaks or volume drops
            # Exit long when: Lips <= Teeth OR Bull Power <= 0 OR Volume not confirmed
            if (lips_aligned[i] <= teeth_aligned[i] or 
                bull_power_aligned[i] <= 0 or 
                not vol_confirm_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position - exit when Alligator alignment breaks or volume drops
            # Exit short when: Lips >= Teeth OR Bear Power >= 0 OR Volume not confirmed
            if (lips_aligned[i] >= teeth_aligned[i] or 
                bear_power_aligned[i] >= 0 or 
                not vol_confirm_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
    
    return signals