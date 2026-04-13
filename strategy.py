#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Williams Alligator trend with 1d volume confirmation
    # Long: Jaw > Teeth > Lips (bullish alignment) + volume > 1.5x 20-period 12h average
    # Short: Jaw < Teeth < Lips (bearish alignment) + volume > 1.5x 20-period 12h average
    # Uses discrete sizing (0.25) to minimize fee drag
    # Target: 12-37 trades/year to stay within 12h optimal range (50-150 total over 4 years)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    median_1d = (df_1d['high'].values + df_1d['low'].values) / 2.0
    
    # Williams Alligator lines (Smoothed Moving Average - SMMA)
    # Jaw: 13-period SMMA, 8 bars ahead
    # Teeth: 8-period SMMA, 5 bars ahead  
    # Lips: 5-period SMMA, 3 bars ahead
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(data[:period])
        # Subsequent values: SMMA = (PREV * (N-1) + CURRENT) / N
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaw = smma(median_1d, 13)
    teeth = smma(median_1d, 8)
    lips = smma(median_1d, 5)
    
    # Shift Alligator lines forward (Jaw +8, Teeth +5, Lips +3)
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_shifted[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_shifted[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_shifted[3:] = lips[:-3]
    
    # Align Alligator lines to 12h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_shifted)
    
    # Volume confirmation: current 12h volume > 1.5x 20-period average
    vol_avg_20 = np.full(n, np.nan)
    for i in range(20, n):
        vol_avg_20[i] = np.mean(volume[i-20:i])
    volume_confirmed = volume > 1.5 * vol_avg_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or 
            np.isnan(teeth_aligned[i]) or
            np.isnan(lips_aligned[i]) or
            np.isnan(volume_confirmed[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = (jaw_aligned[i] > teeth_aligned[i]) and (teeth_aligned[i] > lips_aligned[i])
        bearish_alignment = (jaw_aligned[i] < teeth_aligned[i]) and (teeth_aligned[i] < lips_aligned[i])
        
        # Entry conditions: Alligator alignment + volume confirmation
        enter_long = bullish_alignment and volume_confirmed[i]
        enter_short = bearish_alignment and volume_confirmed[i]
        
        # Exit conditions: loss of alignment
        exit_long = position == 1 and not bullish_alignment
        exit_short = position == -1 and not bearish_alignment
        
        # Execute signals
        if enter_long and position != 1:
            position = 1
            signals[i] = position_size
        elif enter_short and position != -1:
            position = -1
            signals[i] = -position_size
        elif exit_long:
            position = 0
            signals[i] = 0.0
        elif exit_short:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_williams_alligator_volume_v1"
timeframe = "12h"
leverage = 1.0