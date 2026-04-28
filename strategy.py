#!/usr/bin/env python3
# Hypothesis: 4h Williams Alligator with 1d VWAP trend filter and volume confirmation.
# Alligator (Jaw/Teeth/Lips) identifies trend direction and strength; VWAP confirms institutional participation.
# Long when Lips > Teeth > Jaw above VWAP with volume expansion; Short when Lips < Teeth < Jaw below VWAP.
# This trend-following system works in both bull and bear markets by capturing sustained moves with volatility filters.
# Target: 20-40 trades/year to minimize fee drag.

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
    
    # Get daily data for VWAP trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_num = (typical_price_1d * df_1d['volume']).cumsum()
    vwap_den = df_1d['volume'].cumsum()
    vwap_1d = (vwap_num / vwap_den).values
    vwap_1d[vwap_den == 0] = np.nan  # Avoid division by zero
    
    # Align VWAP to 4h timeframe
    vwap_1d_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d)
    
    # Williams Alligator on 4h data
    # Jaw: 13-period SMMA, shifted 8 bars
    # Teeth: 8-period SMMA, shifted 5 bars
    # Lips: 5-period SMMA, shifted 3 bars
    def smma(arr, period):
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        sma = np.convolve(arr, np.ones(period)/period, mode='valid')
        result[period-1:len(sma)+period-1] = sma
        for i in range(len(sma)+period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Shift the lines
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Volume filter: volume > 1.5x 30-period average
    volume_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment
        bullish_alignment = lips_shifted[i] > teeth_shifted[i] > jaw_shifted[i]
        bearish_alignment = lips_shifted[i] < teeth_shifted[i] < jaw_shifted[i]
        
        # VWAP filter
        above_vwap = close[i] > vwap_1d_aligned[i]
        below_vwap = close[i] < vwap_1d_aligned[i]
        
        # Entry conditions
        long_entry = bullish_alignment and above_vwap and volume_filter[i]
        short_entry = bearish_alignment and below_vwap and volume_filter[i]
        
        # Exit conditions: opposite Alligator alignment
        long_exit = bearish_alignment and position == 1
        short_exit = bullish_alignment and position == -1
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit:
            signals[i] = 0.0
            position = 0
        elif short_exit:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_WilliamsAlligator_1dVWAP_VolumeFilter"
timeframe = "4h"
leverage = 1.0