#!/usr/bin/env python3
"""
12h Williams Alligator + 1d EMA50 Trend + Volume Spike
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) identifies trending vs ranging markets.
In 12h timeframe, when Lips cross above Teeth and Jaw with 1d uptrend and volume spike,
it signals strong bullish momentum. Conversely, Lips below Teeth below Jaw with 1d downtrend
and volume spike signals bearish momentum. The 1d EMA50 ensures alignment with higher timeframe trend.
Volume spike confirms participation. Target 12-37 trades/year to minimize fee drag.
Works in both bull (trend continuation) and bear (trend continuation) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars forward
    # Teeth: 8-period SMMA, shifted 5 bars forward  
    # Lips: 5-period SMMA, shifted 3 bars forward
    # SMMA = smoothed moving average (similar to RMA/wilder's)
    
    def smma(arr, period):
        """Smoothed Moving Average (Wilder's)"""
        result = np.full_like(arr, np.nan, dtype=float)
        if len(arr) < period:
            return result
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: (prev*(period-1) + current) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)
    teeth = smma(close, 8)
    lips = smma(close, 5)
    
    # Apply shifts (Jaw shifted 8, Teeth shifted 5, Lips shifted 3)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # First 8 values of jaw_shifted are invalid, etc.
    # We'll check for NaN in the loop
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start index: need enough for all indicators
    start_idx = max(50, 13+8)  # EMA50 warmup + Jaw shift
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or 
            np.isnan(lips_shifted[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        ema_trend = ema_50_aligned[i]
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Volume spike: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma_20 = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20 = np.mean(volume[:i+1])
        volume_spike = curr_volume > 2.0 * vol_ma_20
        
        if position == 0:
            # Long: Lips > Teeth > Jaw (Alligator bullish alignment) AND above 1d EMA50 (uptrend)
            long_condition = (lips_val > teeth_val) and (teeth_val > jaw_val) and (curr_close > ema_trend) and volume_spike
            # Short: Lips < Teeth < Jaw (Alligator bearish alignment) AND below 1d EMA50 (downtrend)
            short_condition = (lips_val < teeth_val) and (teeth_val < jaw_val) and (curr_close < ema_trend) and volume_spike
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
        elif position == 1:
            # Exit long: Alligator loses bullish alignment OR trend breaks
            if not ((lips_val > teeth_val) and (teeth_val > jaw_val)) or curr_close < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator loses bearish alignment OR trend breaks
            if not ((lips_val < teeth_val) and (teeth_val < jaw_val)) or curr_close > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_WilliamsAlligator_1dEMA50_Trend_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0