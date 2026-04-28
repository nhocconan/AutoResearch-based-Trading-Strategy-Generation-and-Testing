#!/usr/bin/env python3
# Hypothesis: 6h Williams Alligator with 1-day ADX trend filter and volume confirmation.
# Williams Alligator (JAW=TEETH=LIPS) uses smoothed moving averages to identify trends.
# In trending markets (ADX>25), the three lines diverge: green (Lips) > red (Teeth) > blue (Jaw) for uptrend,
# and reversed for downtrend. This avoids whipsaws in ranging markets.
# Volume confirmation ensures breakouts have participation.
# Designed for 6h timeframe to target 50-150 total trades over 4 years (12-37/year).
# Works in both bull and bear markets by filtering for strong trends via ADX.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def _smma(array, period):
    """Smoothed Moving Average (SMMA)"""
    if len(array) < period:
        return np.full_like(array, np.nan, dtype=float)
    result = np.full_like(array, np.nan, dtype=float)
    # First value is simple moving average
    result[period-1] = np.mean(array[:period])
    # Subsequent values: SMMA = (prev_smma * (period-1) + current_close) / period
    for i in range(period, len(array)):
        result[i] = (result[i-1] * (period-1) + array[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for ADX calculation
        return np.zeros(n)
    
    # Calculate daily ADX (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = 0
    down_move[0] = 0
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = _smma(tr, 14)
    plus_di_smoothed = _smma(plus_dm, 14)
    minus_di_smoothed = _smma(minus_dm, 14)
    
    # DI values
    plus_di = np.where(atr != 0, plus_di_smoothed / atr * 100, 0)
    minus_di = np.where(atr != 0, minus_di_smoothed / atr * 100, 0)
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
    adx = _smma(dx, 14)
    
    # Align ADX to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Williams Alligator on 6h data
    # Jaw (Blue): 13-period SMMA, shifted 8 bars ahead
    # Teeth (Red): 8-period SMMA, shifted 5 bars ahead  
    # Lips (Green): 5-period SMMA, shifted 3 bars ahead
    jaw = _smma(close, 13)
    teeth = _smma(close, 8)
    lips = _smma(close, 5)
    
    # Apply shifts (future leakage prevention - using only past data)
    # For signal at bar i, we use values that were available at bar i
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill initial shifted values with NaN to avoid look-ahead
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # Volume filter: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Alligator conditions
        # Uptrend: Lips > Teeth > Jaw (all diverging upward)
        # Downtrend: Jaw > Teeth > Lips (all diverging downward)
        lips_val = lips_shifted[i]
        teeth_val = teeth_shifted[i]
        jaw_val = jaw_shifted[i]
        
        # Check for valid alignment (no crossovers)
        uptrend_aligned = lips_val > teeth_val and teeth_val > jaw_val
        downtrend_aligned = jaw_val > teeth_val and teeth_val > lips_val
        
        # Entry conditions with volume confirmation
        long_entry = strong_trend and uptrend_aligned and volume_filter[i]
        short_entry = strong_trend and downtrend_aligned and volume_filter[i]
        
        # Exit conditions: when trend weakens or Alligator lines converge
        # Exit long when trend weakens or Lips crosses below Teeth
        long_exit = (not strong_trend) or (lips_val <= teeth_val) or (position == 1 and close[i] < lips_val)
        # Exit short when trend weakens or Lips crosses above Teeth
        short_exit = (not strong_trend) or (lips_val >= teeth_val) or (position == -1 and close[i] > lips_val)
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
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

name = "6h_WilliamsAlligator_1dADX_TrendFilter_Volume"
timeframe = "6h"
leverage = 1.0