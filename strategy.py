#!/usr/bin/env python3
"""
exp_7375_6h_donchian20_1w_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1w pivot direction and volume confirmation.
Uses weekly pivot points (R3/S3 for mean reversion, R4/S4 for breakout) to capture both reversal and continuation.
In bear markets: fade at R3/S3 (weekly resistance/support). In bull markets: breakout at R4/S4 with volume.
Targets 50-150 trades over 4 years (12-37/year) with signal size 0.25 to manage drawdown.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7375_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # bars for weekly pivot calculation
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 2.0
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 24  # max 4 days (24*6h = 96h = 4d)

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1w for pivot points
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (standard formula)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Pivot point calculations
    pp = (high_1w + low_1w + close_1w) / 3.0
    r1 = 2 * pp - low_1w
    s1 = 2 * pp - high_1w
    r2 = pp + (high_1w - low_1w)
    s2 = pp - (high_1w - low_1w)
    r3 = high_1w + 2 * (pp - low_1w)
    s3 = low_1w - 2 * (high_1w - pp)
    r4 = r3 + (high_1w - low_1w)
    s4 = s3 - (high_1w - low_1w)
    
    # Align to LTF (6h)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
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
    start = max(DONCHIAN_PERIOD, PIVOT_LOOKBACK, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
    for i in range(start, n):
        bars_since_entry += 1
        
        # Skip if HTF data not available
        if np.isnan(pp_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or \
           np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]):
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
        
        # Determine market bias based on price vs weekly pivot
        above_pivot = close[i] > pp_aligned[i]
        below_pivot = close[i] < pp_aligned[i]
        
        # Mean reversion entries at R3/S3 (fade extreme weekly levels)
        mean_revert_long = below_pivot and (close[i] <= s3_aligned[i]) and vol_confirmed
        mean_revert_short = above_pivot and (close[i] >= r3_aligned[i]) and vol_confirmed
        
        # Breakout continuation at R4/S4 (break of extreme weekly levels)
        breakout_long = above_pivot and (close[i] >= r4_aligned[i]) and vol_confirmed
        breakout_short = below_pivot and (close[i] <= s4_aligned[i]) and vol_confirmed
        
        # Donchian breakout with volume confirmation (additional filter)
        donchian_long = (close[i] > highest_high[i]) and vol_confirmed
        donchian_short = (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            # Priority: 1) Mean reversion at R3/S3, 2) Breakout at R4/S4, 3) Donchian breakout
            if mean_revert_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif mean_revert_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif breakout_short:
                signals[i] = -SIGNAL_SIZE
                position = -1
                entry_price = close[i]
                bars_since_entry = 0
            elif donchian_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif donchian_short:
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