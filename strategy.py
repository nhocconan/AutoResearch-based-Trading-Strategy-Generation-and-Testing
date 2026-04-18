#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator + Volume Confirmation + 1d Trend Filter
# Williams Alligator (Jaws, Teeth, Lips) identifies trend direction and strength.
# When all three lines are aligned (bullish or bearish), it indicates a strong trend.
# Volume confirmation ensures institutional participation.
# 1d EMA50 filter ensures alignment with higher timeframe trend to avoid counter-trend trades.
# Works in trending markets (both bull and bear) by capturing sustained moves.
# Target: 20-40 trades/year (80-160 total over 4 years) to minimize fee drag.
name = "6h_WilliamsAlligator_Volume_1dEMA50"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate Williams Alligator on 6h data
    # Jaws (blue line): 13-period SMMA, shifted 8 bars ahead
    # Teeth (red line): 8-period SMMA, shifted 5 bars ahead
    # Lips (green line): 5-period SMMA, shifted 3 bars ahead
    # SMMA = smoothed moving average (similar to EMA but with different smoothing)
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple average
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaws = smma(close, 13)  # 13-period SMMA
    teeth = smma(close, 8)   # 8-period SMMA
    lips = smma(close, 5)    # 5-period SMMA
    
    # Shift the lines as per Williams Alligator specification
    jaws_shifted = np.roll(jaws, 8)   # Shift 8 bars ahead
    teeth_shifted = np.roll(teeth, 5) # Shift 5 bars ahead
    lips_shifted = np.roll(lips, 3)   # Shift 3 bars ahead
    
    # Fill the shifted values at the beginning with NaN
    jaws_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Get 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume spike: current volume > 1.8 * 20-period average volume
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(jaws_shifted[i]) or np.isnan(teeth_shifted[i]) or
            np.isnan(lips_shifted[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        jaw_val = jaws_shifted[i]
        tooth_val = teeth_shifted[i]
        lip_val = lips_shifted[i]
        ema_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Bullish alignment: Lips > Teeth > Jaws AND price above EMA50 AND volume spike
            if lip_val > tooth_val > jaw_val and close_val > ema_val and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Bearish alignment: Lips < Teeth < Jaws AND price below EMA50 AND volume spike
            elif lip_val < tooth_val < jaw_val and close_val < ema_val and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bearish alignment OR price below EMA50 (trend change)
            if lip_val < tooth_val < jaw_val or close_val < ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bullish alignment OR price above EMA50 (trend change)
            if lip_val > tooth_val > jaw_val or close_val > ema_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals