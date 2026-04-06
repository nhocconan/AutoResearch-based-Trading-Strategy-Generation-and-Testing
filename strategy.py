#!/usr/bin/env python3
"""
12h Camarilla Pivot + Volume Spike + Choppiness Regime Filter
Hypothesis: Camarilla pivot levels from 1-day timeframe act as strong support/resistance levels.
Price approaching these levels with volume confirmation and in trending regimes (low chop)
provides high-probability entries. Works in both bull and bear markets by fading extremes
in ranging markets and continuing trends in trending markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_volume_chop"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1-day data for Camarilla pivots (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    range_1d = high_1d - low_1d
    # Resistance levels
    r1 = close_1d + (range_1d * 1.0833)
    r2 = close_1d + (range_1d * 1.1666)
    r3 = close_1d + (range_1d * 1.2500)
    r4 = close_1d + (range_1d * 1.3333)
    # Support levels
    s1 = close_1d - (range_1d * 1.0833)
    s2 = close_1d - (range_1d * 1.1666)
    s3 = close_1d - (range_1d * 1.2500)
    s4 = close_1d - (range_1d * 1.3333)
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day to avoid look-ahead)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # 12h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume spike filter (volume > 2x 24-period average)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    # Choppiness Index (14-period) for regime filter
    # CHOP = 100 * log10(sum(ATR) / (highest high - lowest low)) / log10(n)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    range_14 = highest_high - lowest_low
    range_14 = np.where(range_14 == 0, 1e-10, range_14)
    
    sum_atr = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    chop = 100 * (np.log10(sum_atr) / np.log10(14))
    
    # Regime: CHOP < 40 = trending, CHOP > 60 = ranging
    # We use ranging market for mean reversion at pivot levels
    ranging_market = chop > 60
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 24  # For volume MA and chop calculation
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits
        if position == 1:  # long position
            # Exit: price reaches S1 (strong support) or stoploss
            if (low[i] <= s1_aligned[i] or
                close[i] <= entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R1 (strong resistance) or stoploss
            if (high[i] >= r1_aligned[i] or
                close[i] >= entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging markets only (mean reversion at pivot levels)
            if ranging_market[i]:
                # Long setup: price near S3/S4 with volume spike
                near_support = (low[i] <= s3_aligned[i] * 1.002) or (low[i] <= s4_aligned[i] * 1.002)
                if near_support and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short setup: price near R3/R4 with volume spike
                near_resistance = (high[i] >= r3_aligned[i] * 0.998) or (high[i] >= r4_aligned[i] * 0.998)
                elif near_resistance and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                # In trending markets, stay flat to avoid whipsaws
                signals[i] = 0.0
    
    return signals