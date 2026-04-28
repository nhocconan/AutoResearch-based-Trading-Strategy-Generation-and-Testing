#!/usr/bin/env python3
# Hypothesis: 12h Williams Alligator with 1w ADX trend filter and volume confirmation.
# The Williams Alligator uses three smoothed moving averages (Jaw, Teeth, Lips) to identify trends.
# When the three lines are intertwined (no trend), the market is sleeping; when they diverge, a trend is forming.
# We use 1w ADX > 25 to confirm strong trends and avoid whipsaws in ranging markets.
# Volume confirmation (1.5x 24-period average) ensures breakout validity.
# Designed to work in both bull and bear markets by capturing strong trends while avoiding false signals.
# Targets 50-150 total trades over 4 years (12-37/year) with discrete position sizing to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_smma(data, period):
    """Calculate Smoothed Moving Average (SMMA)"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    smma = np.full_like(data, np.nan, dtype=float)
    smma[period-1] = np.mean(data[:period])
    for i in range(period, len(data)):
        smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    
    # Calculate 1w ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w[1:] - low_1w[1:]
    tr2 = np.abs(high_1w[1:] - close_1w[:-1])
    tr3 = np.abs(low_1w[1:] - close_1w[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value is NaN
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr = calculate_smma(tr[1:], 14)  # Skip first NaN
    atr = np.concatenate([[np.nan], atr])  # Realign
    
    dm_plus_smooth = calculate_smma(dm_plus[1:], 14)
    dm_plus_smooth = np.concatenate([[np.nan], dm_plus_smooth])
    dm_minus_smooth = calculate_smma(dm_minus[1:], 14)
    dm_minus_smooth = np.concatenate([[np.nan], dm_minus_smooth])
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / atr
    di_minus = 100 * dm_minus_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = calculate_smma(dx[2:], 14)  # Skip first two NaN values
    adx = np.concatenate([[np.nan, np.nan], adx])  # Realign
    
    # Align ADX to 12h timeframe (wait for weekly close)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx, additional_delay_bars=0)
    
    # Williams Alligator on 12h data
    # Jaw: 13-period SMMA, shifted 8 bars ahead
    # Teeth: 8-period SMMA, shifted 5 bars ahead
    # Lips: 5-period SMMA, shifted 3 bars ahead
    jaw = calculate_smma(close, 13)
    teeth = calculate_smma(close, 8)
    lips = calculate_smma(close, 5)
    
    # Apply shifts (Jaw: +8, Teeth: +5, Lips: +3)
    jaw_shifted = np.concatenate([np.full(8, np.nan), jaw[:-8]]) if len(jaw) > 8 else np.full_like(jaw, np.nan)
    teeth_shifted = np.concatenate([np.full(5, np.nan), teeth[:-5]]) if len(teeth) > 5 else np.full_like(teeth, np.nan)
    lips_shifted = np.concatenate([np.full(3, np.nan), lips[:-3]]) if len(lips) > 3 else np.full_like(lips, np.nan)
    
    # Volume filter: volume > 1.5x 24-period average (12 days of 12h bars)
    volume_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(24, 13)  # Wait for volume MA and Alligator
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(adx_aligned[i]) or np.isnan(jaw_shifted[i]) or 
            np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or 
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_aligned[i] > 25
        
        # Alligator conditions: 
        # Alligator sleeping (no trend): Jaw, Teeth, Lips intertwined
        # Alligator awake (trend forming): Lines diverging
        # We enter when Alligator awakens in direction of trend
        lips_above_teeth = lips_shifted[i] > teeth_shifted[i]
        teeth_above_jaw = teeth_shifted[i] > jaw_shifted[i]
        lips_below_teeth = lips_shifted[i] < teeth_shifted[i]
        teeth_below_jaw = teeth_shifted[i] < jaw_shifted[i]
        
        # Strong uptrend: Lips > Teeth > Jaw
        alligator_long = lips_above_teeth and teeth_above_jaw
        # Strong downtrend: Lips < Teeth < Jaw
        alligator_short = lips_below_teeth and teeth_below_jaw
        
        # Entry conditions with volume spike confirmation
        long_entry = strong_trend and alligator_long and volume_spike[i]
        short_entry = strong_trend and alligator_short and volume_spike[i]
        
        # Exit conditions: trend weakening or Alligator sleeping
        long_exit = (not strong_trend) or (not alligator_long)
        short_exit = (not strong_trend) or (not alligator_short)
        
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

name = "12h_WilliamsAlligator_1wADX25_VolumeSpike"
timeframe = "12h"
leverage = 1.0