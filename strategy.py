#!/usr/bin/env python3
# 12h_1d_williams_alligator_v1
# Strategy: 12h Williams Alligator with 1d trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Williams Alligator (Jaw, Teeth, Lips) identifies trend phases. 
# Entry when Lips cross above Teeth (bullish) or below Teeth (bearish) with 1d EMA trend alignment and volume confirmation.
# Exits on opposite cross. Designed for low trade frequency (~15-30/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_williams_alligator_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator on 12h data
    # Jaw (blue line): 13-period SMMA, shifted 8 bars forward
    # Teeth (red line): 8-period SMMA, shifted 5 bars forward  
    # Lips (green line): 5-period SMMA, shifted 3 bars forward
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full(len(arr), np.nan)
        result = np.full(len(arr), np.nan)
        # First value is simple moving average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)  # 8-period SMMA
    lips = smma(close, 5)   # 5-period SMMA
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that don't have enough data
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 20-period volume average for confirmation
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(lips_shifted[i]) or np.isnan(teeth_shifted[i]) or 
            np.isnan(jaw_shifted[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg_20[i]
        
        # Alligator signals
        # Bullish: Lips above Teeth (and Teeth above Jaw for strong trend)
        bullish_align = lips_shifted[i] > teeth_shifted[i] and teeth_shifted[i] > jaw_shifted[i]
        # Bearish: Lips below Teeth (and Teeth below Jaw for strong trend)
        bearish_align = lips_shifted[i] < teeth_shifted[i] and teeth_shifted[i] < jaw_shifted[i]
        
        # Cross signals (using previous bar to avoid look-ahead)
        lips_above_teeth_prev = lips_shifted[i-1] > teeth_shifted[i-1]
        lips_below_teeth_prev = lips_shifted[i-1] < teeth_shifted[i-1]
        lips_above_teeth_now = lips_shifted[i] > teeth_shifted[i]
        lips_below_teeth_now = lips_shifted[i] < teeth_shifted[i]
        
        # Bullish cross: Lips cross above Teeth
        bullish_cross = lips_above_teeth_now and not lips_above_teeth_prev
        # Bearish cross: Lips cross below Teeth
        bearish_cross = lips_below_teeth_now and not lips_below_teeth_prev
        
        # 1d EMA trend filter: price above EMA = bullish trend, below = bearish
        trend_bullish = close[i] > ema_50_1d_aligned[i]
        trend_bearish = close[i] < ema_50_1d_aligned[i]
        
        # Entry conditions
        # Long: Bullish cross AND bullish trend AND volume confirmation
        if bullish_cross and trend_bullish and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Bearish cross AND bearish trend AND volume confirmation
        elif bearish_cross and trend_bearish and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Opposite cross
        elif position == 1 and bearish_cross:  # Lips cross below Teeth
            position = 0
            signals[i] = 0.0
        elif position == -1 and bullish_cross:  # Lips cross above Teeth
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals