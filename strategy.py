#!/usr/bin/env python3
"""
exp_6871_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d Camarilla pivot levels and volume confirmation.
In ranging markets (price between daily R3/S3): fade extreme touches of R3/S3 with volume exhaustion.
In trending markets (price breaks R4/S4): continuation breakout trades with volume confirmation.
Uses 1d Camarilla pivot structure to define both mean-reversion and breakout zones.
Designed for 6h timeframe to capture 12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to daily pivot structure.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6871_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.5
VOL_EXIT_THRESHOLD = 0.5
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.0
MAX_HOLD_BARS = 10  # ~2.5 days (6h bars)
PIVOT_LOOKBACK = 1  # use previous day's pivot

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for daily pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Range
    rang = high_1d - low_1d
    
    # Camarilla levels
    r3 = pp + rang * 1.1 / 4.0
    s3 = pp - rang * 1.1 / 4.0
    r4 = pp + rang * 1.1 / 2.0
    s4 = pp - rang * 1.1 / 2.0
    
    # Align to LTF (6h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate LTF indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels
    highest_high = pd.Series(high).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).max().values
    lowest_low = pd.Series(low).rolling(window=DONCHIAN_PERIOD, min_periods=DONCHIAN_PERIOD).min().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(DONCHIAN_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + PIVOT_LOOKBACK + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]):
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
            
        # Volume conditions
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        vol_weak = volume[i] < vol_ma[i] * VOL_EXIT_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Determine market regime based on price vs daily pivot levels
        # Using previous day's levels (already shifted by align_htf_to_ltf)
        price_vs_r3 = close[i] - r3_aligned[i]
        price_vs_s3 = close[i] - s3_aligned[i]
        price_vs_r4 = close[i] - r4_aligned[i]
        price_vs_s4 = close[i] - s4_aligned[i]
        
        # Regime detection
        in_range = (price_vs_r3 <= 0) and (price_vs_s3 >= 0)  # Between R3 and S3
        above_r4 = price_vs_r4 > 0  # Above R4
        below_s4 = price_vs_s4 < 0  # Below S4
        
        # Fade extreme touches in ranging market (mean reversion)
        long_fade = in_range and (price_vs_s3 >= -0.1 * rang[i]) and (price_vs_s3 <= 0.1 * rang[i]) and vol_weak
        short_fade = in_range and (price_vs_r3 <= 0.1 * rang[i]) and (price_vs_r3 >= -0.1 * rang[i]) and vol_weak
        
        # Breakout continuation in trending market
        long_breakout = above_r4 and (close[i] > highest_high[i]) and vol_confirmed
        short_breakout = below_s4 and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if long_fade:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_fade:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif long_breakout:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif short_breakout:
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