#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator system with 12-hour Elder Ray for trend confirmation.
# Williams Alligator: Jaw (13-period, 8-shift), Teeth (8-period, 5-shift), Lips (5-period, 3-shift)
# Elder Ray: Bull Power = High - EMA(13), Bear Power = Low - EMA(13)
# Trend: Alligator aligned (Jaw > Teeth > Lips = bullish, Jaw < Teeth < Lips = bearish)
# Entry: Bull Power > 0 in bullish alignment, Bear Power < 0 in bearish alignment
# Exit: Opposite signal or Alligator crossover
# Works in bull markets by riding trends, works in bear markets by shorting breakdowns.
# Low frequency due to strict alignment requirements. Target: 60-120 trades over 4 years.
name = "exp_14139_6h_alligator_elder_ray_v1"
timeframe = "6h"
leverage = 1.0

def calculate_ema(values, period):
    """Calculate EMA with proper min_periods"""
    return pd.Series(values).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 12h data for Elder Ray (once before loop)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate EMA(13) on 12h close for Elder Ray
    ema_13_12h = calculate_ema(close_12h, 13)
    
    # Calculate Bull Power and Bear Power
    bull_power_12h = high_12h - ema_13_12h
    bear_power_12h = low_12h - ema_13_12h
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_12h, bull_power_12h)
    bear_power_aligned = align_htf_to_ltf(prices, df_12h, bear_power_12h)
    
    # 6h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator components (using SMMA - approximated with EMA for simplicity)
    # Jaw: 13-period EMA, 8 periods into future
    jaw = calculate_ema(close, 13)
    # Teeth: 8-period EMA, 5 periods into future
    teeth = calculate_ema(close, 8)
    # Lips: 5-period EMA, 3 periods into future
    lips = calculate_ema(close, 5)
    
    # Shift components to avoid look-ahead (simulating SMMA shift)
    jaw_shifted = np.roll(jaw, 8)
    teeth_shifted = np.roll(teeth, 5)
    lips_shifted = np.roll(lips, 3)
    # Fill initial values with NaN to simulate proper warmup
    jaw_shifted[:8] = np.nan
    teeth_shifted[:5] = np.nan
    lips_shifted[:3] = np.nan
    
    # ATR for stop loss (14-period)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of shifts and EMA periods)
    start = max(13, 8, 5, 8, 5, 3, 14) + 1
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Alligator alignment
        # Bullish: Jaw > Teeth > Lips
        # Bearish: Jaw < Teeth < Lips
        bullish_alignment = (jaw_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > lips_shifted[i])
        bearish_alignment = (jaw_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < lips_shifted[i])
        
        # Elder Ray signals
        bull_power_signal = bull_power_aligned[i] > 0
        bear_power_signal = bear_power_aligned[i] < 0
        
        # Generate signals
        if position == 0:
            if bullish_alignment and bull_power_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            elif bearish_alignment and bear_power_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or bearish alignment with bear power
            if close[i] <= stop_price or (bearish_alignment and bear_power_signal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or bullish alignment with bull power
            if close[i] >= stop_price or (bullish_alignment and bull_power_signal):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals