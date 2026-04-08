#!/usr/bin/env python3
# [24921] 4h_1d_trix_volume_regime_v1
# Hypothesis: 4-hour TRIX with volume confirmation and 1-day trend filter.
# Long when TRIX crosses above 0, volume > 1.5x average, and price > 1-day EMA100.
# Short when TRIX crosses below 0, volume > 1.5x average, and price < 1-day EMA100.
# Exit when TRIX crosses back across 0 or volume drops below 1.2x average.
# TRIX is a momentum oscillator that filters noise and works in both bull and bear markets.
# Volume confirmation ensures breakouts have conviction.
# Designed to limit trades (~20-30/year) to reduce fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_trix_volume_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate 1-day EMA100 for trend filter
    close_1d = df_1d['close'].values
    ema_100_1d = np.full_like(close_1d, np.nan, dtype=float)
    if len(close_1d) >= 100:
        alpha = 2.0 / (100 + 1)
        ema_100_1d[99] = np.mean(close_1d[:100])
        for i in range(100, len(close_1d)):
            ema_100_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_100_1d[i-1]
    
    # Calculate TRIX (15-period EMA applied 3 times)
    # First EMA
    ema1 = np.full(n, np.nan)
    if len(close) >= 15:
        alpha1 = 2.0 / (15 + 1)
        ema1[14] = np.mean(close[:15])
        for i in range(15, n):
            ema1[i] = alpha1 * close[i] + (1 - alpha1) * ema1[i-1]
    
    # Second EMA of first EMA
    ema2 = np.full(n, np.nan)
    valid_ema1 = ~np.isnan(ema1)
    if np.sum(valid_ema1) >= 15:
        start_idx = np.where(valid_ema1)[0][14]
        alpha2 = 2.0 / (15 + 1)
        ema2[start_idx] = np.mean(ema1[start_idx-14:start_idx+1])
        for i in range(start_idx+1, n):
            if not np.isnan(ema1[i]):
                ema2[i] = alpha2 * ema1[i] + (1 - alpha2) * ema2[i-1]
    
    # Third EMA of second EMA (TRIX)
    ema3 = np.full(n, np.nan)
    valid_ema2 = ~np.isnan(ema2)
    if np.sum(valid_ema2) >= 15:
        start_idx = np.where(valid_ema2)[0][14]
        alpha3 = 2.0 / (15 + 1)
        ema3[start_idx] = np.mean(ema2[start_idx-14:start_idx+1])
        for i in range(start_idx+1, n):
            if not np.isnan(ema2[i]):
                ema3[i] = alpha3 * ema2[i] + (1 - alpha3) * ema3[i-1]
    
    # TRIX = (current EMA3 - previous EMA3) / previous EMA3 * 100
    trix = np.full(n, np.nan)
    for i in range(1, n):
        if not np.isnan(ema3[i]) and not np.isnan(ema3[i-1]) and ema3[i-1] != 0:
            trix[i] = (ema3[i] - ema3[i-1]) / ema3[i-1] * 100
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1-day EMA100 to 4-hour timeframe
    ema_100_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_100_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(ema_100_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        trend_up_1d = price > ema_100_1d_aligned[i]
        
        if position == 1:  # Long
            # Exit: TRIX crosses below 0 or volume drops below 1.2x average
            if trix_now <= 0 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: TRIX crosses above 0 or volume drops below 1.2x average
            if trix_now >= 0 or vol_ratio < 1.2:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: TRIX crosses above 0 with volume expansion and uptrend on 1d
            if trix_now > 0 and trix_prev <= 0 and vol_ratio > 1.5 and trend_up_1d:
                position = 1
                signals[i] = 0.25
            # Enter short: TRIX crosses below 0 with volume expansion and downtrend on 1d
            elif trix_now < 0 and trix_prev >= 0 and vol_ratio > 1.5 and not trend_up_1d:
                position = -1
                signals[i] = -0.25
    
    return signals