#!/usr/bin/env python3

"""
Hypothesis: 6-hour Williams Alligator system with 12-hour trend filter and volume confirmation.
The Williams Alligator (three smoothed moving averages) identifies trend direction and strength.
The 12-hour trend filter (EMA50) ensures trades align with the higher timeframe trend.
Volume spikes confirm institutional participation at trend continuations.
This strategy aims to capture trend continuation moves in both bull and bear markets by
trading when all three Alligator lines are aligned in the same direction with volume confirmation.
Target: 12-37 trades/year per symbol (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smooth_data(data, period):
    """Calculate smoothed moving average (SMMA)"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
    smma = np.full_like(data, np.nan, dtype=float)
    for i in range(len(data)):
        if i < period - 1:
            smma[i] = np.nan
        elif i == period - 1:
            smma[i] = sma.iloc[i]
        else:
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 6h data for Alligator - ONCE before loop
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 6h data
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    jaw = smooth_data(df_6h['close'].values, 13)
    jaw_shifted = np.roll(jaw, 8)
    jaw_shifted[:8] = np.nan
    
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead
    teeth = smooth_data(df_6h['close'].values, 8)
    teeth_shifted = np.roll(teeth, 5)
    teeth_shifted[:5] = np.nan
    
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    lips = smooth_data(df_6h['close'].values, 5)
    lips_shifted = np.roll(lips, 3)
    lips_shifted[:3] = np.nan
    
    # Align Alligator components to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_6h, jaw_shifted)
    teeth_aligned = align_htf_to_ltf(prices, df_6h, teeth_shifted)
    lips_aligned = align_htf_to_ltf(prices, df_6h, lips_shifted)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h volume average (24-period)
    vol_avg_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg_24[i])):
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
        
        # Determine Alligator alignment
        # Bullish alignment: Lips > Teeth > Jaw (green > red > blue)
        # Bearish alignment: Lips < Teeth < Jaw (green < red < blue)
        is_bullish_aligned = lips_aligned[i] > teeth_aligned[i] > jaw_aligned[i]
        is_bearish_aligned = lips_aligned[i] < teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bullish alignment + above 12h EMA + volume spike
            if (is_bullish_aligned and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment + below 12h EMA + volume spike
            elif (is_bearish_aligned and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 2.0 * vol_avg_24[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alligator lines cross or price crosses 12h EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment forms or price crosses below 12h EMA
                if is_bearish_aligned or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish alignment forms or price crosses above 12h EMA
                if is_bullish_aligned or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Williams_Alligator_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0