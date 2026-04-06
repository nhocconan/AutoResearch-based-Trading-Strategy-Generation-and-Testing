#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-week trend filter and volume confirmation.
# The Alligator (Jaw, Teeth, Lips) acts as a trend detector: when lines are intertwined,
# market is ranging (no trade); when diverging, trend is strong. We trade only when
# the Alligator is "awake" (diverging) in alignment with weekly trend, confirmed by volume.
# This avoids whipsaws in ranging markets and catches strong trends in both bull and bear markets.
# Target: 50-150 total trades over 4 years.

name = "exp_13307_6h_alligator_weekly_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
JAW_PERIOD = 13   # Alligator Jaw (blue)
TEETH_PERIOD = 8  # Alligator Teeth (red)
LIPS_PERIOD = 5   # Alligator Lips (green)
JAW_SHIFT = 8
TEETH_SHIFT = 5
LIPS_SHIFT = 3
WEEKLY_EMA_PERIOD = 20
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def calculate_smma(data, period):
    """Smoothed Moving Average (SMMA) used in Alligator"""
    sma = pd.Series(data).rolling(window=period, min_periods=period).mean()
    # Wilder's smoothing: SMMA(t) = (SMMA(t-1)*(period-1) + price(t)) / period
    smma = np.full_like(data, np.nan, dtype=float)
    if len(data) >= period:
        smma[period-1] = sma[period-1]
        for i in range(period, len(data)):
            smma[i] = (smma[i-1] * (period-1) + data[i]) / period
    return smma

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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA for trend filter
    close_1w = df_1w['close'].values
    ema_1w = calculate_ema(close_1w, WEEKLY_EMA_PERIOD)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Alligator components (SMMA with shifts)
    jaw = calculate_smma(close, JAW_PERIOD)
    teeth = calculate_smma(close, TEETH_PERIOD)
    lips = calculate_smma(close, LIPS_PERIOD)
    
    # Apply shifts (Jaw: 8 bars, Teeth: 5 bars, Lips: 3 bars)
    jaw_shifted = np.roll(jaw, JAW_SHIFT)
    teeth_shifted = np.roll(teeth, TEETH_SHIFT)
    lips_shifted = np.roll(lips, LIPS_SHIFT)
    # Set shifted values to NaN where roll creates invalid data
    jaw_shifted[:JAW_SHIFT] = np.nan
    teeth_shifted[:TEETH_SHIFT] = np.nan
    lips_shifted[:LIPS_SHIFT] = np.nan
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(JAW_PERIOD + JAW_SHIFT, TEETH_PERIOD + TEETH_SHIFT, LIPS_PERIOD + LIPS_SHIFT,
                WEEKLY_EMA_PERIOD, VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if any indicator not available
        if (np.isnan(jaw_shifted[i]) or np.isnan(teeth_shifted[i]) or np.isnan(lips_shifted[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(atr[i])):
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
        
        # Trend filter: price above/below weekly EMA
        uptrend = close[i] > ema_1w_aligned[i]
        downtrend = close[i] < ema_1w_aligned[i]
        
        # Alligator "awake" conditions: lines diverging in trend direction
        # For uptrend: Lips > Teeth > Jaw (alligator mouth opening up)
        # For downtrend: Lips < Teeth < Jaw (alligator mouth opening down)
        alligator_awake_up = (lips_shifted[i] > teeth_shifted[i]) and (teeth_shifted[i] > jaw_shifted[i])
        alligator_awake_down = (lips_shifted[i] < teeth_shifted[i]) and (teeth_shifted[i] < jaw_shifted[i])
        
        # Generate signals
        if position == 0:
            if volume_ok and uptrend and alligator_awake_up:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif volume_ok and downtrend and alligator_awake_down:
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