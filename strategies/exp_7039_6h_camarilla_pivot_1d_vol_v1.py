#!/usr/bin/env python3
"""
exp_7039_6h_camarilla_pivot_1d_vol_v1
Hypothesis: 6h Camarilla pivot breakout/fade with 1d volume confirmation. 
In ranging markets (price between daily R3/S3): fade extreme levels (R4/S4) with 1d volume spike.
In trending markets (price outside daily R3/S3): breakout continuation at R4/S4 with 1d volume.
Uses 1d Camarilla pivots calculated from prior 1d bar's H/L/C. Volume confirms institutional interest.
Designed for 6h timeframe to capture ~12-37 trades/year (50-150 total over 4 years) with discrete sizing.
Works in both bull and bear markets by adapting to daily regime (range vs trend).
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7039_6h_camarilla_pivot_1d_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 20  # ~5 months (6h bars)
CHANNEL_LOOKBACK = 10  # for regime detection

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (based on prior 1d bar)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    range_1d = high_1d - low_1d
    pivot = (high_1d + low_1d + close_1d) / 3
    r4 = pivot + (range_1d * 1.1 / 2)
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    s4 = pivot - (range_1d * 1.1 / 2)
    
    # Align to LTF (6h) - note: Camarilla levels are for the 1d bar, so we align with shift(1) via helper
    r4_1d = align_htf_to_ltf(prices, df_1d, r4)
    r3_1d = align_htf_to_ltf(prices, df_1d, r3)
    s3_1d = align_htf_to_ltf(prices, df_1d, s3)
    s4_1d = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r4_1d[i]) or np.isnan(s4_1d[i]):
            signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
            continue
            
        # Check stoploss
        if position == 1:  # long position
            if close[i] <= entry_price - ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
        elif position == -1:  # short position
            if close[i] >= entry_price + ATR_STOP_MULTIPLIER * atr[i]:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
                continue
                
        # Time-based exit
        if position != 0 and bars_since_entry >= MAX_HOLD_BARS:
            signals[i] = 0.0
            position = 0
            bars_since_entry = 0
            continue
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Regime detection: are we in range or trend based on daily levels?
        # Range: price between R3 and S3
        # Trend: price outside R3/S3 (breakout territory)
        in_range = (close[i] >= s3_1d[i]) and (close[i] <= r3_1d[i])
        in_uptrend = close[i] > r3_1d[i]
        in_downtrend = close[i] < s3_1d[i]
        
        # Trading logic based on regime
        long_signal = False
        short_signal = False
        
        if in_range:
            # In range: fade extremes at R4/S4 with volume
            long_signal = (close[i] <= s4_1d[i]) and vol_confirmed  # bounce from S4
            short_signal = (close[i] >= r4_1d[i]) and vol_confirmed  # reject at R4
        elif in_uptrend:
            # In uptrend: breakout continuation at R4
            long_signal = (close[i] > r4_1d[i]) and vol_confirmed
        elif in_downtrend:
            # In downtrend: breakdown continuation at S4
            short_signal = (close[i] < s4_1d[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_signal:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = position * SIGNAL_SIZE
    
    return signals