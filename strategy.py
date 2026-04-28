#!/usr/bin/env python3
"""
12h_Alligator_T123_Gator_Bands
Hypothesis: Williams Alligator (Jaw/Teeth/Lips) on 12h timeframe with Gator Bands and volume confirmation.
Uses Alligator jaws (13-period SMMA shifted 8 bars), teeth (8-period SMMA shifted 5 bars), lips (5-period SMMA shifted 3 bars).
Long when Lips > Teeth > Jaw (bullish alignment), short when Lips < Teeth < Jaw (bearish alignment).
Adds Gator Bands (ATR-based bands around SMMA) for volatility filtering and volume spike confirmation.
Designed for 12h timeframe to capture medium-term trends in both bull and bear markets.
Target: 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(series, period):
    """Smoothed Moving Average (SMMA) - also known as RMA or Wilder's MA"""
    if len(series) < period:
        return np.full_like(series, np.nan, dtype=float)
    result = np.full_like(series, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(series[:period])
    # Subsequent values: (prev * (period-1) + current) / period
    for i in range(period, len(series)):
        result[i] = (result[i-1] * (period-1) + series[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for ATR calculation (for Gator Bands)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate ATR(14) on daily data for Gator Bands
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with index 0
    
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Get 12h data for Alligator components
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate Alligator components (SMMA with specific periods and shifts)
    # Jaw: 13-period SMMA of median price, shifted 8 bars
    median_price_12h = (high_12h := (df_12h['high'].values + df_12h['low'].values) / 2)
    jaw_raw = smma(median_price_12h, 13)
    jaw = np.roll(jaw_raw, 8)  # Shift 8 bars forward
    jaw[:8] = np.nan  # First 8 values invalid after shift
    
    # Teeth: 8-period SMMA of median price, shifted 5 bars
    teeth_raw = smma(median_price_12h, 8)
    teeth = np.roll(teeth_raw, 5)  # Shift 5 bars forward
    teeth[:5] = np.nan  # First 5 values invalid after shift
    
    # Lips: 5-period SMMA of median price, shifted 3 bars
    lips_raw = smma(median_price_12h, 5)
    lips = np.roll(lips_raw, 3)  # Shift 3 bars forward
    lips[:3] = np.nan  # First 3 values invalid after shift
    
    # Align Alligator components to lower timeframe (12h -> 12h is identity but required for consistency)
    jaw_aligned = align_htf_to_ltf(prices, df_12h, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Calculate Gator Bands: ATR-based bands around SMMA (using teeth as base)
    # Using ATR from daily data scaled to 12h (approximate)
    atr_scaled = atr_14 * np.sqrt(12/24)  # Scale daily ATR to 12h approximation
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_scaled)
    
    # Gator Bands: Upper and Lower bands around teeth
    gator_upper = teeth_aligned + (atr_aligned * 1.5)
    gator_lower = teeth_aligned - (atr_aligned * 1.5)
    
    # Volume confirmation: 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(gator_upper[i]) or np.isnan(gator_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Alligator alignment conditions
        bullish_alignment = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        bearish_alignment = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        # Gator Bands conditions (price outside bands indicates strong trend)
        price_above_upper = close[i] > gator_upper[i]
        price_below_lower = close[i] < gator_lower[i]
        
        # Volume confirmation: >1.8x 20-period MA
        vol_confirm = volume[i] > (1.8 * vol_ma_20[i])
        
        # Entry logic
        long_entry = bullish_alignment and price_above_upper and vol_confirm
        short_entry = bearish_alignment and price_below_lower and vol_confirm
        
        # Exit logic: opposite alignment or price returns to teeth (middle line)
        long_exit = (not bullish_alignment) or (close[i] < teeth_aligned[i])
        short_exit = (not bearish_alignment) or (close[i] > teeth_aligned[i])
        
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
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Alligator_T123_Gator_Bands"
timeframe = "12h"
leverage = 1.0