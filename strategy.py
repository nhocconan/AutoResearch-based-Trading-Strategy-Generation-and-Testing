#!/usr/bin/env python3
"""
Hypothesis: 4h Williams Alligator with 1d EMA50 trend filter and volume confirmation.
- Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trend
- Long signal: Lips > Teeth > Jaw (bullish alignment) + price > Lips + volume > 1.5x 20-period avg + price > 1d EMA50
- Short signal: Lips < Teeth < Jaw (bearish alignment) + price < Lips + volume > 1.5x 20-period avg + price < 1d EMA50
- Exit: price crosses back below/above Teeth or opposite Alligator alignment
- 1d EMA50 ensures alignment with daily trend to avoid counter-trend trades
- Volume confirmation reduces false signals in low-participation moves
- Target: 20-50 trades/year (75-200 total over 4 years) to minimize fee drag on 4h timeframe
- Williams Alligator is effective in both trending and ranging markets, providing clear trend direction
"""

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
    
    # Williams Alligator: three smoothed moving averages
    # Jaw (blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (green): 5-period SMMA, shifted 3 bars ahead
    def smma(data, period):
        """Smoothed Moving Average"""
        sma = np.full_like(data, np.nan, dtype=float)
        if len(data) >= period:
            sma[period-1] = np.mean(data[:period])
            for i in range(period, len(data)):
                sma[i] = (sma[i-1] * (period-1) + data[i]) / period
        return sma
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines as per Alligator specification
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume confirmation: > 1.5x 20-period average (spike filter)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Load 1d data ONCE before loop for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 13)  # Need 50 for 1d EMA, 20 for volume MA, 13 for Alligator
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # Volume spike confirmation (> 1.5x average)
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long entry: bullish alignment + price > Lips + volume spike + price > 1d EMA50
            if bullish_alignment and volume_spike and close[i] > lips_shifted[i] and close[i] > ema_50_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish alignment + price < Lips + volume spike + price < 1d EMA50
            elif bearish_alignment and volume_spike and close[i] < lips_shifted[i] and close[i] < ema_50_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Teeth or bearish alignment
            if close[i] < teeth_shifted[i] or not bullish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Teeth or bullish alignment
            if close[i] > teeth_shifted[i] or not bearish_alignment:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike"
timeframe = "4h"
leverage = 1.0