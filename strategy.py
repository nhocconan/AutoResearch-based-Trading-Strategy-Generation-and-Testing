#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams Alligator with 1d trend filter and volume confirmation.
# Long when Alligator jaw < teeth < lips (bullish alignment) AND 1d EMA50 rising AND volume > 1.5x 20-period average.
# Short when Alligator jaw > teeth > lips (bearish alignment) AND 1d EMA50 falling AND volume > 1.5x 20-period average.
# Exit when Alligator lines re-cross (jaws cross teeth) or price crosses 8-period EMA.
# The Alligator identifies trend phases via smoothed SMAs. The 1d EMA50 ensures alignment with daily trend.
# Volume confirmation filters weak breakouts. This combination should work in both bull and bear markets
# by following the dominant trend while avoiding choppy periods.
# Target: 50-120 total trades over 4 years (12-30/year) to minimize fee drag.

name = "6h_WilliamsAlligator_1dEMA50_Volume"
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
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
    # SMMA = Smoothed Moving Average (similar to Wilder's smoothing)
    def smma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close, 13)  # Blue line
    teeth = smma(close, 8)  # Red line
    lips = smma(close, 5)   # Green line
    
    # Shift the lines as per Alligator definition
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    
    # Set NaN for shifted positions
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d EMA50 direction
    ema50_rising = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_falling = np.zeros_like(ema50_1d_aligned, dtype=bool)
    ema50_rising[1:] = ema50_1d_aligned[1:] > ema50_1d_aligned[:-1]
    ema50_falling[1:] = ema50_1d_aligned[1:] < ema50_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Sufficient warmup for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Check for NaN values
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema50_rising[i]) or np.isnan(ema50_falling[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = (jaw_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < lips_shifted[i])
        bearish_alignment = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        
        if position == 0:
            # Long conditions: bullish alignment, 1d EMA50 rising, volume filter
            long_cond = bullish_alignment and ema50_rising[i] and volume_filter[i]
            # Short conditions: bearish alignment, 1d EMA50 falling, volume filter
            short_cond = bearish_alignment and ema50_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish alignment (jaws cross teeth) or price crosses below 8-period EMA
            ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
            exit_cond = bearish_alignment or (close[i] < ema8[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish alignment (jaws cross teeth) or price crosses above 8-period EMA
            ema8 = pd.Series(close).ewm(span=8, adjust=False, min_periods=8).mean().values
            exit_cond = bullish_alignment or (close[i] > ema8[i])
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals