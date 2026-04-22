#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Alligator with 12-hour trend filter and volume confirmation.
The Alligator (Jaw/Teeth/Lips) identifies trend direction and strength.
The 12-hour trend filter ensures trades align with the higher timeframe trend.
Volume spikes confirm participation at Alligator alignment points.
This strategy aims to catch strong trending moves in both bull and bear markets by
trading when all three Alligator lines are aligned in the same direction.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_alligator(high, low, jaw=13, teeth=8, lips=5):
    """Calculate Alligator lines: Jaw (13), Teeth (8), Lips (5) SMAs of median price"""
    median_price = (high + low) / 2
    
    # Jaw: 13-period SMA shifted 8 bars ahead
    jaw_line = (pd.Series(median_price).rolling(window=jaw, min_periods=jaw).mean()).shift(8)
    
    # Teeth: 8-period SMA shifted 5 bars ahead
    teeth_line = (pd.Series(median_price).rolling(window=teeth, min_periods=teeth).mean()).shift(5)
    
    # Lips: 5-period SMA shifted 3 bars ahead
    lips_line = (pd.Series(median_price).rolling(window=lips, min_periods=lips).mean()).shift(3)
    
    return jaw_line.values, teeth_line, lips_line

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h Alligator data - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    # Calculate Alligator on 6h data
    jaw_6h, teeth_6h, lips_6h = calculate_alligator(
        df_6h['high'].values, df_6h['low'].values
    )
    
    # Align Alligator lines to 6h timeframe
    jaw_6h_aligned = align_htf_to_ltf(prices, df_6h, jaw_6h)
    teeth_6h_aligned = align_htf_to_ltf(prices, df_6h, teeth_6h)
    lips_6h_aligned = align_htf_to_ltf(prices, df_6h, lips_6h)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend filter (21-period)
    close_12h = df_12h['close'].values
    ema_21_12h = pd.Series(close_12h).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Calculate 6h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_6h_aligned[i]) or np.isnan(teeth_6h_aligned[i]) or 
            np.isnan(lips_6h_aligned[i]) or np.isnan(ema_21_12h_aligned[i]) or
            np.isnan(vol_avg_24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Check Alligator alignment
        # Bullish alignment: Lips > Teeth > Jaw (all pointing up)
        bullish_aligned = (lips_6h_aligned[i] > teeth_6h_aligned[i] > jaw_6h_aligned[i])
        # Bearish alignment: Lips < Teeth < Jaw (all pointing down)
        bearish_aligned = (lips_6h_aligned[i] < teeth_6h_aligned[i] < jaw_6h_aligned[i])
        
        if position == 0:
            # Long: Bullish alignment + above 12h EMA + volume spike
            if (bullish_aligned and 
                close[i] > ema_21_12h_aligned[i] and 
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + below 12h EMA + volume spike
            elif (bearish_aligned and 
                  close[i] < ema_21_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross or price crosses 12h EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment or price below 12h EMA
                if bearish_aligned or close[i] < ema_21_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish alignment or price above 12h EMA
                if bullish_aligned or close[i] > ema_21_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Alligator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0