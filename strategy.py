#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d Williams Alligator (Jaw/Teeth/Lips) + volume confirmation
# Donchian breakouts capture momentum; 1d Alligator shows trend alignment across timeframes
# Volume confirmation ensures breakout authenticity with conviction
# Works in bull/bear: Alligator adapts to higher timeframe trend strength and direction
# Target: 75-200 total trades over 4 years (19-50/year) with discrete sizing 0.25-0.30

name = "4h_1d_alligator_breakout_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Alligator calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 1d timeframe
    # Jaw (blue): 13-period SMMA, offset 8 bars
    # Teeth (red): 8-period SMMA, offset 5 bars  
    # Lips (green): 5-period SMMA, offset 3 bars
    close_1d = df_1d['close'].values
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        result = np.full(len(arr), np.nan)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_CLOSE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # 13-period SMMA
    teeth = smma(close_1d, 8)  # 8-period SMMA
    lips = smma(close_1d, 5)   # 5-period SMMA
    
    # Apply offsets: Jaw offset 8, Teeth offset 5, Lips offset 3
    jaw_offset = np.full_like(jaw, np.nan)
    teeth_offset = np.full_like(teeth, np.nan)
    lips_offset = np.full_like(lips, np.nan)
    
    if len(jaw) > 8:
        jaw_offset[8:] = jaw[:-8]
    if len(teeth) > 5:
        teeth_offset[5:] = teeth[:-5]
    if len(lips) > 3:
        lips_offset[3:] = lips[:-3]
    
    # Align Alligator lines to 4h timeframe (wait for daily close)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw_offset)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth_offset)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips_offset)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * avg_volume[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = bullish alignment
        # Alligator alignment: Lips < Teeth < Jaw = bearish alignment
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price < Donchian low OR Alligator loses bullish alignment
            if close[i] < donchian_low[i] or not bullish_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price > Donchian high OR Alligator loses bearish alignment
            if close[i] > donchian_high[i] or not bearish_aligned:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic with volume confirmation and Donchian breakout + Alligator filter
            if volume_confirmed:
                # Long entry: price > Donchian high AND bullish Alligator alignment
                if close[i] > donchian_high[i] and bullish_aligned:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price < Donchian low AND bearish Alligator alignment
                elif close[i] < donchian_low[i] and bearish_aligned:
                    position = -1
                    signals[i] = -0.25
    
    return signals