#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with volume spike and 1d choppiness regime filter.
Long when price breaks above Camarilla R3 AND volume > 2.0x average AND CHOP > 61.8 (ranging).
Short when price breaks below Camarilla S3 AND volume > 2.0x average AND CHOP > 61.8.
Exit when price reverts to Camarilla HLC midpoint OR CHOP < 38.2 (trending).
Uses 4h for Camarilla calculation and 1d for CHOP filter to capture mean reversion in ranging markets.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise support/resistance,
volume confirmation filters weak breakouts, CHOP filter ensures we only trade in ranging conditions.
Works in ranging markets (2025-2026 bear/range) and avoids trending markets that cause false breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla levels on 4h timeframe (previous period)
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.1*(high-low), etc.
    # We use the previous 4h bar's high/low/close to calculate levels for current bar
    prev_high_4h = np.roll(high_4h, 1)
    prev_low_4h = np.roll(low_4h, 1)
    prev_close_4h = np.roll(close_4h, 1)
    
    # First bar: use current values (will be refined with more data)
    prev_high_4h[0] = high_4h[0]
    prev_low_4h[0] = low_4h[0]
    prev_close_4h[0] = close_4h[0]
    
    rang = prev_high_4h - prev_low_4h
    r3 = prev_close_4h + 1.1 * rang
    s3 = prev_close_4h - 1.1 * rang
    hlc = (prev_high_4h + prev_low_4h + prev_close_4h) / 3.0
    
    # Get 1d data for Choppiness Index filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index on 1d timeframe (14-period)
    high_1d_series = pd.Series(high_1d)
    low_1d_series = pd.Series(low_1d)
    close_1d_series = pd.Series(close_1d)
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # ATR (14-period)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh = high_1d_series.rolling(window=14, min_periods=14).max().values
    ll = low_1d_series.rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index: 100 * log10(sum(tr)/abs(hh-ll)) / log10(14)
    # Avoid division by zero
    price_range = np.abs(hh - ll)
    chop = np.where(price_range != 0, 
                    100 * np.log10(sum_tr / price_range) / np.log10(14), 
                    50.0)  # neutral when range is zero
    
    # Align 4h Camarilla to 4h timeframe (no alignment needed for same timeframe)
    r3_aligned = r3
    s3_aligned = s3
    hlc_aligned = hlc
    
    # Align 1d CHOP to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume average (20-period) on 4h
    volume_4h = df_4h['volume'].values
    volume_ma = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(hlc_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        r3_val = r3_aligned[i]
        s3_val = s3_aligned[i]
        hlc_val = hlc_aligned[i]
        chop_val = chop_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > R3 AND volume > 2.0x avg AND CHOP > 61.8 (ranging)
            if price > r3_val and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < S3 AND volume > 2.0x avg AND CHOP > 61.8 (ranging)
            elif price < s3_val and vol > 2.0 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < HLC midpoint OR CHOP < 38.2 (trending)
            if price < hlc_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > HLC midpoint OR CHOP < 38.2 (trending)
            if price > hlc_val or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_CamarillaR3S3_Volume_CHOP_Filter"
timeframe = "4h"
leverage = 1.0