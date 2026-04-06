#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator (Jaws, Teeth, Lips) combined with 1-day trend filter.
# Alligator identifies trend phases: when lines are intertwined (sleeping) = range,
# when diverging (awakening) = trend. Uses 1-day EMA to filter direction.
# Works in bull markets (long when bullish alignment + above daily EMA)
# and bear markets (short when bearish alignment + below daily EMA).
# Target: 50-150 total trades over 4 years (12-37/year).

name = "exp_13491_6h_alligator_1d_ema_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
JAWS_PERIOD = 13   # Blue line
TEETH_PERIOD = 8   # Red line
LIPS_PERIOD = 5    # Green line
EMA_PERIOD = 21    # Daily EMA for trend filter
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5

def calculate_smma(values, period):
    """Calculate Smoothed Moving Average (SMMA) - Wilder's smoothing"""
    sma = pd.Series(values).rolling(window=period, min_periods=period).mean()
    # First value is SMA, then smoothed
    smma = np.full_like(values, np.nan, dtype=float)
    if len(values) >= period:
        smma[period-1] = sma.iloc[period-1]
        for i in range(period, len(values)):
            smma[i] = (smma[i-1] * (period-1) + values[i]) / period
    return smma

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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily EMA for trend filter
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=EMA_PERIOD, adjust=False, min_periods=EMA_PERIOD).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams Alligator (using SMMA)
    jaws = calculate_smma(high, JAWS_PERIOD)  # Typically uses median price, but high for sensitivity
    teeth = calculate_smma(high, TEETH_PERIOD)
    lips = calculate_smma(high, LIPS_PERIOD)
    
    # ATR for stoploss
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(JAWS_PERIOD, TEETH_PERIOD, LIPS_PERIOD, EMA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if indicators not available
        if (np.isnan(jaws[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i])):
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
        
        # Alligator alignment conditions
        # Bullish alignment: Lips > Teeth > Jaws (green above red above blue)
        bullish_alignment = lips[i] > teeth[i] and teeth[i] > jaws[i]
        # Bearish alignment: Jaws > Teeth > Lips (blue above red above green)
        bearish_alignment = jaws[i] > teeth[i] and teeth[i] > lips[i]
        
        # Trend filter from daily EMA
        above_daily_ema = close[i] > ema_1d_aligned[i]
        below_daily_ema = close[i] < ema_1d_aligned[i]
        
        # Entry signals
        if position == 0:
            if bullish_alignment and above_daily_ema:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_alignment and below_daily_ema:
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