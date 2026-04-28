#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d Elder Ray bear power filter and volume confirmation.
# Enter long when Alligator jaws (13-period SMMA) cross above teeth (8-period SMMA) and lips (5-period SMMA),
# 1d Elder Ray bear power is negative (bullish), and volume > 1.5x 20-bar average.
# Enter short when Alligator jaws cross below teeth and lips, 1d Elder Ray bull power is positive (bearish),
# and volume > 1.5x 20-bar average.
# Exit when Alligator jaws cross back through teeth or lips in opposite direction.
# Uses discrete position sizing (0.25) to balance return and fee drag.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid excessive fee drag.
# Williams Alligator identifies trend initiation; Elder Ray confirms institutional bias;
# Volume spike confirms participation. Effective in both bull and bear markets via trend following.

name = "12h_Williams_Alligator_1dElderRay_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Alligator calculation
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Williams Alligator (SMMA = smoothed moving average)
    close_12h = df_12h['close'].values
    
    # SMMA calculation: first value = SMA, subsequent = (prev*period-1 + current)/period
    def smma(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) < period:
            return result
        # First value is SMA
        result[period-1] = np.mean(data[:period])
        # Subsequent values
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    jaws = smma(close_12h, 13)   # Blue line
    teeth = smma(close_12h, 8)   # Red line
    lips = smma(close_12h, 5)    # Green line
    
    # Align Alligator lines to 12h timeframe (already aligned as we're using 12h data)
    # But we need to align to primary timeframe (12h) - since we're using 12h data on 12h timeframe,
    # the values are already aligned. We'll use them directly.
    jaws_aligned = jaws
    teeth_aligned = teeth
    lips_aligned = lips
    
    # Get 1d data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_1d = df_1d['close'].values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema_13
    bear_power = df_1d['low'].values - ema_13
    
    # Align Elder Ray to 12h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(jaws_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Alligator conditions: jaws above teeth and lips = uptrend, jaws below = downtrend
        alligator_long = jaws_aligned[i] > teeth_aligned[i] and jaws_aligned[i] > lips_aligned[i]
        alligator_short = jaws_aligned[i] < teeth_aligned[i] and jaws_aligned[i] < lips_aligned[i]
        
        # Elder Ray conditions: bear power negative = bullish, bull power positive = bearish
        elder_bullish = bear_power_aligned[i] < 0  # Bear power negative = bullish
        elder_bearish = bull_power_aligned[i] > 0  # Bull power positive = bearish
        
        # Exit conditions: Alligator jaws cross back through teeth or lips
        exit_long = jaws_aligned[i] < teeth_aligned[i] or jaws_aligned[i] < lips_aligned[i]
        exit_short = jaws_aligned[i] > teeth_aligned[i] or jaws_aligned[i] > lips_aligned[i]
        
        # Handle entries and exits
        if alligator_long and elder_bullish and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif alligator_short and elder_bearish and vol_confirm and position >= 0:
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