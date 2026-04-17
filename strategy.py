#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla Pivot R1/S1 Breakout with Volume Spike and 1d Choppiness Filter.
Long when price breaks above R1 with volume > 1.5x 20-period average AND 1d Choppiness Index > 61.8 (ranging market).
Short when price breaks below S1 with volume > 1.5x 20-period average AND 1d Choppiness Index > 61.8.
Exit when price reverts to the 1d EMA50 or opposite pivot level is touched.
Uses 1d for Choppiness Index and EMA50 regime filter, 4h for pivot calculation and entry.
Target: 75-200 total trades over 4 years (19-50/year). Camarilla pivots provide precise intraday support/resistance,
volume confirmation filters weak breakouts, and chop filter ensures we only trade in ranging markets where mean reversion works.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index and EMA50 regime filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for regime filter
    close_1d_series = pd.Series(close_1d)
    ema50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d Choppiness Index (CHOP)
    def calculate_chop(high_arr, low_arr, close_arr, period=14):
        """Calculate Choppiness Index: higher values = more ranging, lower = more trending"""
        atr_sum = np.zeros(len(close_arr))
        true_range = np.zeros(len(close_arr))
        
        for i in range(1, len(close_arr)):
            hl = high_arr[i] - low_arr[i]
            hc = abs(high_arr[i] - close_arr[i-1])
            lc = abs(low_arr[i] - close_arr[i-1])
            true_range[i] = max(hl, hc, lc)
        
        # Calculate ATR using Wilder's smoothing (equivalent to RMA)
        atr = np.zeros(len(close_arr))
        atr[period] = np.mean(true_range[1:period+1])  # seed
        
        for i in range(period+1, len(close_arr)):
            atr[i] = (atr[i-1] * (period-1) + true_range[i]) / period
        
        # Calculate highest high and lowest low over period
        hh = np.zeros(len(close_arr))
        ll = np.zeros(len(close_arr))
        
        for i in range(period, len(close_arr)):
            hh[i] = np.max(high_arr[i-period+1:i+1])
            ll[i] = np.min(low_arr[i-period+1:i+1])
        
        # Chop formula: 100 * log10(sum(atr) / (hh - ll)) / log10(period)
        chop = np.full(len(close_arr), 50.0)  # default neutral
        for i in range(period, len(close_arr)):
            if hh[i] > ll[i]:  # avoid division by zero
                sum_atr = np.sum(atr[i-period+1:i+1])
                chop[i] = 100 * np.log10(sum_atr / (hh[i] - ll[i])) / np.log10(period)
        
        return chop
    
    chop_1d = calculate_chop(high_1d, low_1d, close_1d, 14)
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 4h bar
    # R1 = Close + 1.1*(High-Low)/12
    # S1 = Close - 1.1*(High-Low)/12
    camarilla_r1_4h = np.full(len(close_4h), np.nan)
    camarilla_s1_4h = np.full(len(close_4h), np.nan)
    
    for i in range(1, len(close_4h)):
        high_val = high_4h[i-1]
        low_val = low_4h[i-1]
        close_val = close_4h[i-1]
        camarilla_r1_4h[i] = close_val + 1.1 * (high_val - low_val) / 12
        camarilla_s1_4h[i] = close_val - 1.1 * (high_val - low_val) / 12
    
    # Align 1d indicators to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Align 4h Camarilla levels to 4h timeframe (already aligned but ensure proper shifting)
    camarilla_r1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1_4h)
    camarilla_s1_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1_4h)
    
    # Calculate volume spike filter (volume > 1.5x 20-period average)
    volume_ma = np.zeros(len(volume))
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = max(100, 50)  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or \
           np.isnan(camarilla_r1_4h_aligned[i]) or np.isnan(camarilla_s1_4h_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_spike = volume_spike[i]
        r1 = camarilla_r1_4h_aligned[i]
        s1 = camarilla_s1_4h_aligned[i]
        ema50 = ema50_1d_aligned[i]
        chop = chop_1d_aligned[i]
        
        # Only trade in ranging markets (Choppiness Index > 61.8)
        in_range = chop > 61.8
        
        if position == 0:
            # Long: price breaks above R1 with volume spike AND in ranging market
            if price > r1 and vol_spike and in_range:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike AND in ranging market
            elif price < s1 and vol_spike and in_range:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price reverts to EMA50 OR touches S1 (mean reversion complete)
            if price < ema50 or price < s1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price reverts to EMA50 OR touches R1 (mean reversion complete)
            if price > ema50 or price > r1:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_ChopFilter"
timeframe = "4h"
leverage = 1.0