#!/usr/bin/env python3
"""
exp_6979_6h_camarilla1d_pivot_v1
Hypothesis: 6h Camarilla pivot reversals from 1d levels with volume confirmation.
In ranging markets: fade extreme levels (R3/S3, R4/S4) for mean reversion.
In trending markets: breakout continuation beyond R4/S4 with volume.
Uses 1d Camarilla pivots calculated from prior day's OHLC to avoid look-ahead.
Designed for low trade frequency (12-37/year) with discrete sizing to minimize fees.
Works in both bull and bear markets by adapting to volatility regimes.
"""

from mtf_data import get_htf_data, align_htf_to_ltf
import numpy as np
import pandas as pd

name = "exp_6979_6h_camarilla1d_pivot_v1"
timeframe = "6h"
leverage = 1.0

# Parameters
SIGNAL_SIZE = 0.25
ATR_PERIOD = 14
ATR_STOP_MULTIPLIER = 2.5
VOL_MA_PERIOD = 20
VOL_BASE_THRESHOLD = 1.8
MAX_HOLD_BARS = 24  # 4 days max hold (6h bars)
CHOP_PERIOD = 14
CHOP_THRESHOLD = 50  # <50 = trending, >50 = ranging

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop - using 1d for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d Camarilla pivot levels (using prior day's OHLC to avoid look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Prior day's values (shift by 1 to avoid look-ahead)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_1d = prev_high - prev_low
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    r4 = pivot + (range_1d * 1.1 / 2.0)
    s4 = pivot - (range_1d * 1.1 / 2.0)
    
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
    
    # ATR for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean().values
    
    # Volume MA for confirmation
    vol_ma = pd.Series(volume).rolling(window=VOL_MA_PERIOD, min_periods=VOL_MA_PERIOD).mean().values
    
    # Choppiness Index for regime detection
    highest_high_h = pd.Series(high).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).max().values
    lowest_low_l = pd.Series(low).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).min().values
    atr_sum = pd.Series(atr).rolling(window=CHOP_PERIOD, min_periods=CHOP_PERIOD).sum().values
    chop = 100 * np.log10(atr_sum / np.log10(CHOP_PERIOD) / (highest_high_h - lowest_low_l))
    chop = np.where((highest_high_h - lowest_low_l) > 0, chop, 50)  # avoid div by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = max(CHOP_PERIOD, VOL_MA_PERIOD, ATR_PERIOD) + 1
    
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
            
        # Volume confirmation
        vol_confirmed = volume[i] > vol_ma[i] * VOL_BASE_THRESHOLD if not np.isnan(vol_ma[i]) else False
        
        # Regime detection
        is_ranging = chop[i] > CHOP_THRESHOLD
        is_trending = chop[i] <= CHOP_THRESHOLD
        
        # Initialize signal
        signals[i] = position * SIGNAL_SIZE if position != 0 else 0.0
        
        # Enter new positions only if flat
        if position == 0:
            if is_ranging:
                # Ranging market: fade extreme levels
                long_fade = close[i] <= s3_aligned[i] and vol_confirmed
                short_fade = close[i] >= r3_aligned[i] and vol_confirmed
                
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
            else:
                # Trending market: breakout continuation
                long_breakout = close[i] > r4_aligned[i] and vol_confirmed
                short_breakout = close[i] < s4_aligned[i] and vol_confirmed
                
                if long_breakout:
                    signals[i] = SIGNAL_SIZE
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                elif short_breakout:
                    signals[i] = -SIGNAL_SIZE
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
    
    return signals