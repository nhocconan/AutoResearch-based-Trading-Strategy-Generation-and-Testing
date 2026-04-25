#!/usr/bin/env python3
"""
12h Williams Alligator + Volume Spike + ATR Filter
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trend direction and strength. 
Breakouts above/below the Alligator with volume confirmation capture strong moves in both bull and bear markets.
ATR filter ensures sufficient volatility. Designed for 12h timeframe to minimize fee drag while maintaining edge.
Target: 12-30 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator: Jaw (13,8), Teeth (8,5), Lips (5,3) - SMMA
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        sma = np.mean(arr[:period])
        result[period-1] = sma
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Align Alligator lines to 12h timeframe (use previous completed 1d bar)
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # ATR filter: ensure sufficient volatility
    tr = np.maximum(np.maximum(high - low, np.abs(high - np.roll(close, 1))), np.abs(low - np.roll(close, 1)))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_filter = atr_14 > (atr_ma * 0.8)  # Trade when volatility is above 80% of its 50-period MA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for all indicators
    start_idx = max(50, 20, 14, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr_filter[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        atr_ok = atr_filter[i]
        
        # Alligator alignment conditions
        # Bullish: Lips > Teeth > Jaw (green alignment)
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish: Lips < Teeth < Jaw (red alignment)
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Price position relative to Alligator
        price_above_alligator = curr_close > lips_aligned[i] and curr_close > teeth_aligned[i] and curr_close > jaw_aligned[i]
        price_below_alligator = curr_close < lips_aligned[i] and curr_close < teeth_aligned[i] and curr_close < jaw_aligned[i]
        
        if position == 0:
            # Look for entry signals - require: Alligator alignment + price breakout + volume + ATR filter
            long_entry = bullish_aligned and price_above_alligator and vol_spike and atr_ok
            short_entry = bearish_aligned and price_below_alligator and vol_spike and atr_ok
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price closes below Teeth (trend weakening)
            if curr_close < teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price closes above Teeth (trend weakening)
            if curr_close > teeth_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_Trend_VolumeSpike_ATRFilter"
timeframe = "12h"
leverage = 1.0