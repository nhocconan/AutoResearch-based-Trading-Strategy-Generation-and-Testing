#!/usr/bin/env python3
# 6h_1d_camarilla_pullback_v1
# Strategy: 6-hour pullback to Camarilla pivot levels with 1-day trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In trending markets (identified by 1d EMA50), price pulls back to Camarilla
# pivot levels (S3/S4 for longs, R3/R4 for shorts) offering high-probability entries.
# Works in bull markets via long pullbacks to S3/S4 in uptrends, and in bear markets via
# short pullbacks to R3/R4 in downtrends. Uses volume confirmation to avoid false signals.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # S1 = C - (Range * 1.1 / 12)
    # S2 = C - (Range * 1.1 / 6)
    # S3 = C - (Range * 1.1 / 4)
    # S4 = C - (Range * 1.1 / 2)
    # R1 = C + (Range * 1.1 / 12)
    # R2 = C + (Range * 1.1 / 6)
    # R3 = C + (Range * 1.1 / 4)
    # R4 = C + (Range * 1.1 / 2)
    
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    s1_1d = close_1d - (range_1d * 1.1 / 12)
    s2_1d = close_1d - (range_1d * 1.1 / 6)
    s3_1d = close_1d - (range_1d * 1.1 / 4)
    s4_1d = close_1d - (range_1d * 1.1 / 2)
    
    r1_1d = close_1d + (range_1d * 1.1 / 12)
    r2_1d = close_1d + (range_1d * 1.1 / 6)
    r3_1d = close_1d + (range_1d * 1.1 / 4)
    r4_1d = close_1d + (range_1d * 1.1 / 2)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 6h timeframe (wait for daily close)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(s3_1d_aligned[i]) or np.isnan(s4_1d_aligned[i]) or 
            np.isnan(r3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Long when price pulls back to S3/S4 in uptrend with volume confirmation
        long_signal = ((price_close <= s3_1d_aligned[i] * 1.002) and 
                       (price_close >= s4_1d_aligned[i] * 0.998)) and \
                      uptrend_1d and vol_spike[i]
        
        # Short when price pulls back to R3/R4 in downtrend with volume confirmation
        short_signal = ((price_close >= r3_1d_aligned[i] * 0.998) and 
                        (price_close <= r4_1d_aligned[i] * 1.002)) and \
                       downtrend_1d and vol_spike[i]
        
        # Exit when price moves back toward the pivot (mean reversion completion)
        exit_long = position == 1 and (price_close > pivot_1d_aligned[i] * 1.001)
        exit_short = position == -1 and (price_close < pivot_1d_aligned[i] * 0.999)
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals

# Align pivot for exit condition
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)