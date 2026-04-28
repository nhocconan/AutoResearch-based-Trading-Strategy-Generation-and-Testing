#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray trend filter and volume spike confirmation.
# Enter long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA),
# 1d Elder Ray Bull Power > 0, and volume > 2.0x 20-bar average.
# Enter short when jaws cross below teeth, 1d Elder Ray Bear Power < 0, and volume > 2.0x 20-bar average.
# Exit when Alligator jaws cross back in opposite direction or Elder Ray power reverses.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 75-150 total trades over 4 years (19-37/year) to avoid excessive fee churn.
# Williams Alligator identifies trend initiation; Elder Ray confirms bull/bear power;
# Volume spike validates breakout strength. Works in bull/bear via trend-following logic.

name = "12h_Williams_Alligator_1dElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def smma(data, period):
    """Smoothed Moving Average (SMMA) - Williams Alligator uses SMMA"""
    if len(data) < period:
        return np.full_like(data, np.nan, dtype=float)
    result = np.full_like(data, np.nan, dtype=float)
    # First value is simple SMA
    result[period-1] = np.mean(data[:period])
    # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_DATA) / period
    for i in range(period, len(data)):
        result[i] = (result[i-1] * (period-1) + data[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Get 12h data for Williams Alligator
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 13:
        return np.zeros(n)
    
    # Calculate Williams Alligator (all SMMA)
    close_12h = df_12h['close'].values
    jaws = smma(close_12h, 13)   # 13-period SMMA (blue line)
    teeth = smma(close_12h, 8)   # 8-period SMMA (red line)
    lips = smma(close_12h, 5)    # 5-period SMMA (green line)
    
    # Align Alligator lines to 12h timeframe (no additional delay needed for SMMA)
    jaws_aligned = align_htf_to_ltf(prices, df_12h, jaws)
    teeth_aligned = align_htf_to_ltf(prices, df_12h, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_12h, lips)
    
    # Volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 13)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Williams Alligator signals: jaws cross teeth
        jaw_teeth_cross_up = jaws_aligned[i] > teeth_aligned[i] and jaws_aligned[i-1] <= teeth_aligned[i-1]
        jaw_teeth_cross_down = jaws_aligned[i] < teeth_aligned[i] and jaws_aligned[i-1] >= teeth_aligned[i-1]
        
        # Elder Ray trend: bull/bear power
        bull_power_positive = bull_power_aligned[i] > 0
        bear_power_negative = bear_power_aligned[i] < 0
        
        # Exit conditions: Alligator cross reverses or Elder Ray power reverses
        exit_long = (jaws_aligned[i] < teeth_aligned[i]) or (bull_power_aligned[i] <= 0)
        exit_short = (jaws_aligned[i] > teeth_aligned[i]) or (bear_power_aligned[i] >= 0)
        
        # Handle entries and exits
        if jaw_teeth_cross_up and bull_power_positive and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif jaw_teeth_cross_down and bear_power_negative and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
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