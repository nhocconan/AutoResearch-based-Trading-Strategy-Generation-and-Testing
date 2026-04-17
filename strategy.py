#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R1/S1 breakout with volume spike and 1d choppiness regime filter.
Long when price breaks above Camarilla R1 AND volume > 1.8x 20-period average AND daily CHOP > 61.8 (ranging market for mean reversion).
Short when price breaks below Camarilla S1 AND volume > 1.8x average AND daily CHOP > 61.8.
Exit when price reverts to Camarilla H5/L5 level OR daily CHOP < 38.2 (trending market).
Uses 4h for price/volume, 1d for CHOP filter to avoid whipsaw in strong trends.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla levels provide precise intraday support/resistance,
volume confirmation reduces fakeouts, choppiness filter ensures we only trade in ranging markets where mean reversion works.
Works in bull markets (buys dips to S1 in ranging uptrends) and bear markets (sells rallies to R1 in ranging downtrends).
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
    
    # Get 4h data for Camarilla calculation and volume
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate Camarilla levels for 4h timeframe (based on previous day's OHLC)
    # Camarilla levels use previous period's range
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    # H5 = close + (high - low) * 1.1/2
    # L5 = close - (high - low) * 1.1/2
    # H3 = close + (high - low) * 1.1/4
    # L3 = close - (high - low) * 1.1/4
    # R3 = close + (high - low) * 1.1/6
    # S3 = close - (high - low) * 1.1/6
    # R4 = close + (high - low) * 1.1/8
    # S4 = close - (high - low) * 1.1/8
    
    # Shift to get previous bar's OHLC for current bar's levels
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]  # first period
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    range_4h = prev_high - prev_low
    camarilla_r1 = prev_close + range_4h * 1.1 / 12
    camarilla_s1 = prev_close - range_4h * 1.1 / 12
    camarilla_h5 = prev_close + range_4h * 1.1 / 2
    camarilla_l5 = prev_close - range_4h * 1.1 / 2
    
    # Calculate volume average (20-period) on 4h
    volume_series = pd.Series(volume_4h)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for choppiness index
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Choppiness Index (CHOP) on 1d timeframe (14-period)
    # CHOP = 100 * log10(sum(ATR) / (n * (highest_high - lowest_low))) / log10(n)
    # where ATR = True Range
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = 100 * np.log10(atr_sum / (14 * (highest_high - lowest_low))) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((highest_high - lowest_low) != 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    # Align 4h Camarilla levels, volume MA, and 1d CHOP to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_h5)
    camarilla_l5_aligned = align_htf_to_ltf(prices, df_4h, camarilla_l5)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(camarilla_h5_aligned[i]) or np.isnan(camarilla_l5_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        r1 = camarilla_r1_aligned[i]
        s1 = camarilla_s1_aligned[i]
        h5 = camarilla_h5_aligned[i]
        l5 = camarilla_l5_aligned[i]
        vol_ma = volume_ma_aligned[i]
        chop_val = chop_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price > Camarilla R1 AND volume > 1.8x avg AND daily CHOP > 61.8 (ranging market)
            if price > r1 and vol > 1.8 * vol_ma and chop_val > 61.8:
                signals[i] = 0.25
                position = 1
            # Short: price < Camarilla S1 AND volume > 1.8x avg AND daily CHOP > 61.8 (ranging market)
            elif price < s1 and vol > 1.8 * vol_ma and chop_val > 61.8:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < Camarilla H5 OR daily CHOP < 38.2 (trending market)
            if price < h5 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > Camarilla L5 OR daily CHOP < 38.2 (trending market)
            if price > l5 or chop_val < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Volume_CHOP_Filter"
timeframe = "4h"
leverage = 1.0