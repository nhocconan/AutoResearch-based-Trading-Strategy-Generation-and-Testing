#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day Trend Filter and Volume Confirmation.
Long when Alligator jaws are above teeth and lips (bullish alignment) during 1-day uptrend with volume spike.
Short when jaws are below teeth and lips (bearish alignment) during 1-day downtrend with volume spike.
Exit when alignment breaks or trend reverses.
Williams Alligator uses smoothed moving averages (SMMA) of median price to identify trends.
Designed for moderate trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend and Alligator alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - also called Wilder's Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator: (high + low) / 2
    median_price = (high + low) / 2
    
    # Williams Alligator parameters (standard: 13, 8, 5 with offsets 8, 5, 3)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Calculate SMMA lines
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Apply shifts (forward shift - looking into future, so we need to handle carefully)
    # For Alligator, the lines are shifted forward, meaning we use past values to represent current
    jaw_shifted = np.full_like(jaw, np.nan)
    teeth_shifted = np.full_like(teeth, np.nan)
    lips_shifted = np.full_like(lips, np.nan)
    
    if len(jaw) > jaw_shift:
        jaw_shifted[jaw_shift:] = jaw[:-jaw_shift]
    if len(teeth) > teeth_shift:
        teeth_shifted[teeth_shift:] = teeth[:-teeth_shift]
    if len(lips) > lips_shift:
        lips_shifted[lips_shift:] = lips[:-lips_shift]
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup period: need enough data for all indicators
    warmup = max(jaw_period + jaw_shift, teeth_period + teeth_shift, lips_period + lips_shift, 20) + 5
    
    for i in range(warmup, n):
        # Skip if data not ready
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator alignment conditions
        jaw_val = jaw_shifted[i]
        teeth_val = teeth_shifted[i]
        lips_val = lips_shifted[i]
        
        # Bullish alignment: jaws > teeth > lips
        bullish_alignment = jaw_val > teeth_val and teeth_val > lips_val
        # Bearish alignment: jaws < teeth < lips
        bearish_alignment = jaw_val < teeth_val and teeth_val < lips_val
        
        if position == 0:
            # Long: bullish alignment + 1d uptrend + volume spike
            if bullish_alignment and ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: bearish alignment + 1d downtrend + volume spike
            elif bearish_alignment and ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: alignment breaks or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: bullish alignment breaks or 1d trend turns down
                if not bullish_alignment or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: bearish alignment breaks or 1d trend turns up
                if not bearish_alignment or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Williams_Alligator_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0