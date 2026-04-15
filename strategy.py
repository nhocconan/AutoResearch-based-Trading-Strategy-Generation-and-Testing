#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Volume Confirmation
# Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
# In trending markets: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
# In ranging markets: lines intertwine
# Volume confirmation: current volume > 1.5 * 20-period average volume
# Only trade when Alligator shows clear alignment AND volume confirms
# Designed for low trade frequency with strong trend signals
# Works in bull markets (follow alignment) and avoids whipsaws in ranging markets (no trade when intertwined)

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data once
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Smoothed Moving Average (SMMA) - same as Wilder's smoothing
    def smma(data, period):
        if len(data) < period:
            return np.full_like(data, np.nan, dtype=float)
        result = np.full_like(data, np.nan, dtype=float)
        # First value is simple moving average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (prev_smma * (period-1) + current_value) / period
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    # Williams Alligator lines on 12h
    jaw = smma(close_12h, 13)  # 13-period SMMA
    teeth = smma(close_12h, 8)  # 8-period SMMA
    lips = smma(close_12h, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator specification
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # Volume confirmation on 1d
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    # 20-period average volume
    vol_ma = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    # Volume ratio: current volume / average volume
    vol_ratio = np.where(vol_ma > 0, volume_1d / vol_ma, 0)
    
    # Align indicators to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    vol_ratio_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(lips_aligned[i]) or np.isnan(vol_ratio_aligned[i])):
            continue
        
        # Alligator alignment: Lips > Teeth > Jaw (bullish) or Lips < Teeth < Jaw (bearish)
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Volume confirmation: volume > 1.5 * average volume
        volume_confirmed = vol_ratio_aligned[i] > 1.5
        
        # Only trade when both conditions are met
        if bullish_alignment and volume_confirmed and position <= 0:
            position = 1
            signals[i] = position_size
        elif bearish_alignment and volume_confirmed and position >= 0:
            position = -1
            signals[i] = -position_size
        # Exit when alignment breaks (lines intertwine)
        elif position == 1 and not bullish_alignment:
            position = 0
            signals[i] = 0.0
        elif position == -1 and not bearish_alignment:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "12h_1d_WilliamsAlligator_Volume"
timeframe = "12h"
leverage = 1.0