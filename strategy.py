#!/usr/bin/env python3
"""
Experiment #7691: 6-hour Camarilla pivot from 1-day with volume confirmation.
Hypothesis: Fade at R3/S3 levels when price reverts to mean, breakout continuation at R4/S4.
Uses 1-day Camarilla levels calculated from prior day's range, applied to 6-hour timeframe.
Works in ranging markets (mean reversion at R3/S3) and trending markets (breakout at R4/S4).
Target: 50-150 total trades over 4 years.
"""

from mtf_data import get_alt_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7691_6h_camarilla_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
CAMARILLA_MULT = 1.1  # Standard Camarilla multiplier
VOLUME_MA_PERIOD = 20
VOLUME_THRESHOLD = 1.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1-day Camarilla levels from previous day's range
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    # R4 = Close + 1.1 * Range * 1.1/2 (actually R4 = Close + Range * 1.1)
    # Standard: R4 = Close + Range * 1.1, R3 = Close + Range * 1.1/2
    r4 = close_1d + (range_1d * CAMARILLA_MULT)
    r3 = close_1d + (range_1d * CAMARILLA_MULT / 2)
    s3 = close_1d - (range_1d * CAMARILLA_MULT / 2)
    s4 = close_1d - (range_1d * CAMARILLA_MULT)
    
    # Align to 6h timeframe (shifted by 1 day for look-ahead bias prevention)
    r4_6h = align_htf_to_ltf(prices, df_1d, r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Volume moving average
    volume_ma = pd.Series(volume).rolling(window=VOLUME_MA_PERIOD, min_periods=VOLUME_MA_PERIOD).mean().values
    
    # ATR for risk management
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period
    start = max(VOLUME_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        # Skip if HTF data not available
        if np.isnan(r4_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or np.isnan(s4_6h[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
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
        volume_confirmed = volume[i] > (volume_ma[i] * VOLUME_THRESHOLD) if not np.isnan(volume_ma[i]) else False
        
        # Fade at R3/S3 (mean reversion)
        fade_long = (close[i] <= s3_6h[i]) and volume_confirmed
        fade_short = (close[i] >= r3_6h[i]) and volume_confirmed
        
        # Breakout continuation at R4/S4
        breakout_long = (close[i] >= r4_6h[i]) and volume_confirmed
        breakout_short = (close[i] <= s4_6h[i]) and volume_confirmed
        
        # Generate signals
        if position == 0:
            if fade_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif fade_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (ATR_STOP_MULTIPLIER * atr[i])
            elif breakout_short:
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