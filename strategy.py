#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_12h_camarilla_breakout_v1
# Camarilla pivot levels from 12h combined with volume confirmation and 4h trend filter.
# Captures breakouts from key pivot levels in trending markets with volume confirmation.
# Designed to work in both bull and bear markets by filtering trades with trend.
# Target: 20-40 trades/year to avoid fee drag.

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    pivot = (high_12h + low_12h + close_12h) / 3
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r4 = close_12h + range_12h * 1.1 / 2
    r3 = close_12h + range_12h * 1.1 / 4
    r2 = close_12h + range_12h * 1.1 / 6
    r1 = close_12h + range_12h * 1.1 / 12
    s1 = close_12h - range_12h * 1.1 / 12
    s2 = close_12h - range_12h * 1.1 / 6
    s3 = close_12h - range_12h * 1.1 / 4
    s4 = close_12h - range_12h * 1.1 / 2
    
    # Combine levels into arrays
    camarilla_high = np.maximum.reduce([r1, r2, r3, r4])
    camarilla_low = np.minimum.reduce([s1, s2, s3, s4])
    
    # Align Camarilla levels to 4h timeframe
    camarilla_high_aligned = align_htf_to_ltf(prices, df_12h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_12h, camarilla_low)
    
    # 4h trend filter: EMA50
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if Camarilla levels not ready
        if np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long signal: price breaks above Camarilla resistance with volume and uptrend
        long_signal = (
            close[i] > camarilla_high_aligned[i] and
            vol_confirm[i] and
            close[i] > ema_50[i]
        )
        
        # Short signal: price breaks below Camarilla support with volume and downtrend
        short_signal = (
            close[i] < camarilla_low_aligned[i] and
            vol_confirm[i] and
            close[i] < ema_50[i]
        )
        
        # Exit conditions: opposite Camarilla level touch
        exit_long = close[i] < camarilla_low_aligned[i]
        exit_short = close[i] > camarilla_high_aligned[i]
        
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals