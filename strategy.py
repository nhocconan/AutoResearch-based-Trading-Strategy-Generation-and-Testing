#!/usr/bin/env python3
# 4h_12h_camarilla_breakout_v1
# Strategy: 4-hour Camarilla pivot breakout with 12-hour trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Uses daily resampled Camarilla pivot levels from 12h timeframe (R3/S3 for reversals, R4/S4 for breakouts)
# filtered by 12h EMA50 trend and volume spikes. Works in both bull and bear markets by
# aligning with higher timeframe trend while capturing intraday momentum bursts.
# Targets 75-200 trades over 4 years to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_12h_camarilla_breakout_v1"
timeframe = "4h"
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
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h OHLC for Camarilla pivots and EMA50
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla pivot levels for previous 12h period
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    r4_12h = close_12h + range_12h * 1.1 / 2.0
    r3_12h = close_12h + range_12h * 1.1 / 4.0
    s3_12h = close_12h - range_12h * 1.1 / 4.0
    s4_12h = close_12h - range_12h * 1.1 / 2.0
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h data to 4h timeframe (wait for 12h close)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, r4_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, s4_12h)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    pivot_12h_aligned = align_htf_to_ltf(prices, df_12h, pivot_12h)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(s3_12h_aligned[i]) or np.isnan(s4_12h_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 12h EMA50
        uptrend_12h = price_close > ema_50_12h_aligned[i]
        downtrend_12h = price_close < ema_50_12h_aligned[i]
        
        # Camarilla breakout signals (using previous 12h level)
        breakout_up = price_close > r4_12h_aligned[i]   # Break above R4
        breakdown_down = price_close < s4_12h_aligned[i]  # Break below S4
        reverse_at_r3 = price_close < r3_12h_aligned[i] and price_close > r3_12h_aligned[i-1]  # Reject at R3
        reverse_at_s3 = price_close > s3_12h_aligned[i] and price_close < s3_12h_aligned[i-1]  # Reject at S3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above R4 with volume in uptrend OR rejection at R3 with volume in uptrend
        long_signal = (breakout_up and vol_confirmed and uptrend_12h) or \
                      (reverse_at_r3 and vol_confirmed and uptrend_12h)
        
        # Short: Break below S4 with volume in downtrend OR rejection at S3 with volume in downtrend
        short_signal = (breakdown_down and vol_confirmed and downtrend_12h) or \
                       (reverse_at_s3 and vol_confirmed and downtrend_12h)
        
        # Exit when price returns to the 12h pivot level or opposite Camarilla level
        exit_long = position == 1 and (price_close < pivot_12h_aligned[i] or 
                                       price_close < s3_12h_aligned[i])
        exit_short = position == -1 and (price_close > pivot_12h_aligned[i] or 
                                         price_close > r3_12h_aligned[i])
        
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