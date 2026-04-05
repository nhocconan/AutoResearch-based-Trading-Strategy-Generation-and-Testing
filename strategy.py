#!/usr/bin/env python3
"""
exp_7287_6h_donchian20_1d_pivot_vol_v1
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction filter and volume confirmation.
In trending markets (price > weekly pivot): continuation breakouts in breakout direction.
In ranging markets (price near pivot): mean reversion at Donchian extremes with volume confirmation.
Uses 1d data to calculate weekly pivot points (standard formula: PP=(H+L+C)/3, R1=2*PP-L, S1=2*PP-H, etc.).
Designed for 6h timeframe to capture swings with ~12-37 trades/year (50-150 total over 4 years).
Works in both bull and bear markets by adapting to pivot-defined trend regime.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_7287_6h_donchian20_1d_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
DONCHIAN_PERIOD = 20
PIVOT_LOOKBACK = 5  # days to calculate weekly pivot from prior week
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
MAX_HOLD_BARS = 8  # ~48 hours

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for weekly pivot
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by PIVOT_LOOKBACK days to use prior week's data (avoiding look-ahead)
    high_shifted = np.roll(high_1d, PIVOT_LOOKBACK)
    low_shifted = np.roll(low_1d, PIVOT_LOOKBACK)
    close_shifted = np.roll(close_1d, PIVOT_LOOKBACK)
    
    # Set first PIVOT_LOOKBACK values to NaN (invalid prior week data)
    high_shifted[:PIVOT_LOOKBACK] = np.nan
    low_shifted[:PIVOT_LOOKBACK] = np.nan
    close_shifted[:PIVOT_LOOKBACK] = np.nan
    
    # Calculate pivot points: PP = (H+L+C)/3
    pivot_point = (high_shifted + low_shifted + close_shifted) / 3.0
    # Resistance and support levels
    r1 = 2 * pivot_point - low_shifted
    s1 = 2 * pivot_point - high_shifted
    r2 = pivot_point + (high_shifted - low_shifted)
    s2 = pivot_point - (high_shifted - low_shifted)
    r3 = high_shifted + 2 * (pivot_point - low_shifted)
    s3 = low_shifted - 2 * (high_shifted - pivot_point)
    
    # Align to LTF (6h)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_point)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
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
        if np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]):
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
        
        # Determine market regime based on pivot
        above_pivot = close[i] > pivot_aligned[i]
        below_pivot = close[i] < pivot_aligned[i]
        near_pivot = np.abs(close[i] - pivot_aligned[i]) < (0.5 * atr[i])  # Within 0.5 ATR of pivot
        
        # Fade at extremes in ranging market (near pivot)
        fade_long = near_pivot and (close[i] <= lowest_low[i]) and vol_confirmed
        fade_short = near_pivot and (close[i] >= highest_high[i]) and vol_confirmed
        
        # Continuation breakouts in trending market (above/below pivot)
        continuation_long = above_pivot and (close[i] > highest_high[i]) and vol_confirmed
        continuation_short = below_pivot and (close[i] < lowest_low[i]) and vol_confirmed
        
        # Enter new positions only if flat
        if position == 0:
            if fade_long or continuation_long:
                signals[i] = SIGNAL_SIZE
                position = 1
                entry_price = close[i]
                bars_since_entry = 0
            elif fade_short or continuation_short:
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