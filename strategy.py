#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams Alligator + 1d EMA50 Trend + Volume Spike
# Long when price > Alligator Jaw AND Alligator Teeth > Alligator Lips AND price > 1d EMA50 AND volume > 2.0x 20-bar avg
# Short when price < Alligator Jaw AND Alligator Teeth < Alligator Lips AND price < 1d EMA50 AND volume > 2.0x 20-bar avg
# Exit when Alligator lines converge (Teeth crosses Jaw) OR price crosses 1d EMA50 in opposite direction
# Uses discrete position sizing (0.25) to reduce fee drag. Target: 19-50 trades/year on 4h timeframe.
# Williams Alligator identifies trend phases, 1d EMA50 filters counter-trend moves,
# volume confirmation ensures breakout strength. This combination should work in both bull and bear markets
# by only taking trades aligned with the higher timeframe trend and confirming with volume.

name = "4h_WilliamsAlligator_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 1d data
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    # Align EMA50 to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Williams Alligator on 4h data (Smoothed Moving Average with period 5, 8, 13)
    # Jaw (blue): 13-period SMMA shifted 8 bars ahead
    # Teeth (red): 8-period SMMA shifted 5 bars ahead  
    # Lips (green): 5-period SMMA shifted 3 bars ahead
    def smma(values, period):
        """Smoothed Moving Average"""
        if len(values) < period:
            return np.full(len(values), np.nan)
        result = np.full(len(values), np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(values[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(values)):
            result[i] = (result[i-1] * (period-1) + values[i]) / period
        return result
    
    jaw = smma(close, 13)  # Jaw period
    teeth = smma(close, 8)  # Teeth period
    lips = smma(close, 5)   # Lips period
    
    # Shift as per Alligator definition: Jaw shifted 8, Teeth shifted 5, Lips shifted 3
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted values that went out of bounds
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50, 13)  # volume MA, EMA50, and Alligator warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_conf = volume_confirm[i]
        curr_ema50 = ema_50_1d_aligned[i]
        curr_jaw = jaw_shifted[i]
        curr_teeth = teeth_shifted[i]
        curr_lips = lips_shifted[i]
        curr_close = close[i]
        
        # Handle exits and position management
        if position == 1:  # Long position
            # Exit: Alligator lines converge (Teeth crosses below Jaw) OR price crosses below 1d EMA50
            if curr_teeth < curr_jaw or curr_close < curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Alligator lines converge (Teeth crosses above Jaw) OR price crosses above 1d EMA50
            if curr_teeth > curr_jaw or curr_close > curr_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                
        else:  # Flat - look for new entries
            # Long when price > Jaw AND Teeth > Lips AND price > 1d EMA50 AND volume confirmation
            if curr_close > curr_jaw and curr_teeth > curr_lips and curr_close > curr_ema50 and vol_conf:
                signals[i] = 0.25
                position = 1
            # Short when price < Jaw AND Teeth < Lips AND price < 1d EMA50 AND volume confirmation
            elif curr_close < curr_jaw and curr_teeth < curr_lips and curr_close < curr_ema50 and vol_conf:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
    
    return signals