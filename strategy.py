#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator + Elder Ray (Bull/Bear Power) with volume confirmation.
# Alligator identifies trend direction using smoothed medians (Jaw, Teeth, Lips).
# Elder Ray measures bull/bear power as price deviation from EMA13.
# Long when: price > Teeth, Bull Power > 0, volume > 1.5x MA.
# Short when: price < Teeth, Bear Power < 0, volume > 1.5x MA.
# Works in bull markets (riding uptrends) and bear markets (riding downtrends).
# Target: 50-150 total trades over 4 years with low turnover.

name = "exp_13371_6h_alligator_elder_ray_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAW_PERIOD = 13   # Smoothed Median (13-period SMMA, shifted 8 bars)
ALLIGATOR_TEETH_PERIOD = 8  # Smoothed Median (8-period SMMA, shifted 5 bars)
ALLIGATOR_LIPS_PERIOD = 5   # Smoothed Median (5-period SMMA, shifted 3 bars)
ELDER_RAY_EMA_PERIOD = 13   # EMA for Bull/Bear Power
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: (prev*(period-1) + current) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def calculate_ema(close, period):
    """Calculate EMA"""
    return pd.Series(close).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    """Calculate ATR using Wilder's smoothing"""
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for Alligator and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Alligator components (Jaw, Teeth, Lips) from daily median prices
    median_price = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw_raw = smma(median_price, ALLIGATOR_JAW_PERIOD)
    teeth_raw = smma(median_price, ALLIGATOR_TEETH_PERIOD)
    lips_raw = smma(median_price, ALLIGATOR_LIPS_PERIOD)
    
    # Shift components as per Alligator definition
    jaw = np.roll(jaw_raw, ALLIGATOR_JAW_PERIOD + 5)  # Shifted 8+5? Actually 8 bars for jaw
    teeth = np.roll(teeth_raw, ALLIGATOR_TEETH_PERIOD + 3)  # Shifted 5+3? Actually 5 bars for teeth
    lips = np.roll(lips_raw, ALLIGATOR_LIPS_PERIOD + 2)   # Shifted 3+2? Actually 3 bars for lips
    # Correct shifts: Jaw 8, Teeth 5, Lips 3
    jaw = np.roll(jaw_raw, 8)
    teeth = np.roll(teeth_raw, 5)
    lips = np.roll(lips_raw, 3)
    
    # Align to 6h timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Calculate EMA13 for Elder Ray
    close_1d = df_1d['close'].values
    ema13 = calculate_ema(close_1d, ELDER_RAY_EMA_PERIOD)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAW_PERIOD + 8, ALLIGATOR_TEETH_PERIOD + 5, ALLIGATOR_LIPS_PERIOD + 3,
                ELDER_RAY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema13_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = position * SIGNAL_SIZE
            else:
                signals[i] = 0.0
            continue
        
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # short position
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Volume confirmation
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD)
        
        # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
        bull_power = high[i] - ema13_aligned[i]
        bear_power = low[i] - ema13_aligned[i]
        
        # Alligator signals: price relationship to Teeth (middle line)
        price_above_teeth = close[i] > teeth_aligned[i]
        price_below_teeth = close[i] < teeth_aligned[i]
        
        # Entry conditions
        long_signal = volume_ok and price_above_teeth and (bull_power > 0)
        short_signal = volume_ok and price_below_teeth and (bear_power < 0)
        
        # Generate signals
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            signals[i] = SIGNAL_SIZE
        elif position == -1:
            signals[i] = -SIGNAL_SIZE
    
    return signals