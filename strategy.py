#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Williams Alligator with 1-week trend filter and volume confirmation.
# The Williams Alligator (Jaws/Teeth/Lips) identifies trends when lines are separated and aligned.
# Weekly trend filter ensures trades align with higher timeframe momentum.
# Volume confirmation filters false signals. Works in both bull/bear by capturing established trends.
# Target: 75-200 total trades over 4 years (19-50/year) to balance opportunity and cost.

name = "alligator_6h_1w_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
ALLIGATOR_JAWS = 13   # Smoothed MA (13-period, 8-shift)
ALLIGATOR_TEETH = 8   # Smoothed MA (8-period, 5-shift)
ALLIGATOR_LIPS = 5    # Smoothed MA (5-period, 3-shift)
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def smooth(sma_vals, period):
    """Apply additional smoothing to SMAs (Williams Alligator method)"""
    return pd.Series(sma_vals).rolling(window=period, min_periods=period).mean().values

def calculate_ma(close, period):
    """Calculate simple moving average"""
    return pd.Series(close).rolling(window=period, min_periods=period).mean().values

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
    ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Calculate 6h indicators
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams Alligator components
    jaws_raw = calculate_ma(close, ALLIGATOR_JAWS)
    teeth_raw = calculate_ma(close, ALLIGATOR_TEETH)
    lips_raw = calculate_ma(close, ALLIGATOR_LIPS)
    
    # Apply smoothing (shifted)
    jaws = smooth(jaws_raw, ALLIGATOR_JAWS)
    teeth = smooth(teeth_raw, ALLIGATOR_TEETH)
    lips = smooth(lips_raw, ALLIGATOR_LIPS)
    
    # Shift for proper alignment (Williams Alligator uses future-shifted SMAs)
    jaws = np.roll(jaws, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Volume MA
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR
    atr = calculate_atr(high, low, close, ATR_PERIOD)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(ALLIGATOR_JAWS + 8, ALLIGATOR_TEETH + 5, ALLIGATOR_LIPS + 3, 
                VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if weekly EMA not available
        if np.isnan(ema_1w_aligned[i]):
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
        volume_ok = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Weekly trend filter
        uptrend_weekly = close[i] > ema_1w_aligned[i]
        downtrend_weekly = close[i] < ema_1w_aligned[i]
        
        # Alligator signals: aligned and separated
        # Bullish: Lips > Teeth > Jaws (all ascending)
        bullish_aligned = (lips[i] > teeth[i] > jaws[i]) and \
                         (lips[i] > lips[i-1]) and (teeth[i] > teeth[i-1]) and (jaws[i] > jaws[i-1])
        # Bearish: Jaws > Teeth > Lips (all descending)
        bearish_aligned = (jaws[i] > teeth[i] > lips[i]) and \
                         (jaws[i] < jaws[i-1]) and (teeth[i] < teeth[i-1]) and (lips[i] < lips[i-1])
        
        # Generate signals
        if position == 0:
            if bullish_aligned and volume_ok and uptrend_weekly:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif bearish_aligned and volume_ok and downtrend_weekly:
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