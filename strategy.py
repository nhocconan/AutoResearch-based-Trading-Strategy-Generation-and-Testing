#!/usr/bin/env python3
# 12h_1w_camarilla_breakout_volume_trend_v1
# Strategy: 12-hour Camarilla pivot breakout with 1-week trend filter and volume confirmation
# Timeframe: 12h
# Leverage: 1.0
# Hypothesis: Uses weekly Camarilla pivot levels (R3/S3 for reversals, R4/S4 for breakouts)
# filtered by 1-week EMA50 trend and volume spikes. Designed to capture multi-day momentum
# bursts while filtering counter-trend noise. Targets 50-150 total trades over 4 years.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1w_camarilla_breakout_volume_trend_v1"
timeframe = "12h"
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
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w OHLC for Camarilla pivots and EMA50
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla pivot levels for previous week
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    r4_1w = close_1w + range_1w * 1.1 / 2.0
    r3_1w = close_1w + range_1w * 1.1 / 4.0
    s3_1w = close_1w - range_1w * 1.1 / 4.0
    s4_1w = close_1w - range_1w * 1.1 / 2.0
    
    # 1w EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w data to 12h timeframe (wait for weekly close)
    r3_1w_aligned = align_htf_to_ltf(prices, df_1w, r3_1w)
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s3_1w_aligned = align_htf_to_ltf(prices, df_1w, s3_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r3_1w_aligned[i]) or np.isnan(r4_1w_aligned[i]) or 
            np.isnan(s3_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1w EMA50
        uptrend_1w = price_close > ema_50_1w_aligned[i]
        downtrend_1w = price_close < ema_50_1w_aligned[i]
        
        # Camarilla breakout signals (using previous week's levels)
        breakout_up = price_close > r4_1w_aligned[i]   # Break above R4
        breakdown_down = price_close < s4_1w_aligned[i]  # Break below S4
        reverse_at_r3 = price_close < r3_1w_aligned[i] and price_close > r3_1w_aligned[i-1]  # Reject at R3
        reverse_at_s3 = price_close > s3_1w_aligned[i] and price_close < s3_1w_aligned[i-1]  # Reject at S3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above R4 with volume in uptrend OR rejection at R3 with volume in uptrend
        long_signal = (breakout_up and vol_confirmed and uptrend_1w) or \
                      (reverse_at_r3 and vol_confirmed and uptrend_1w)
        
        # Short: Break below S4 with volume in downtrend OR rejection at S3 with volume in downtrend
        short_signal = (breakdown_down and vol_confirmed and downtrend_1w) or \
                       (reverse_at_s3 and vol_confirmed and downtrend_1w)
        
        # Exit when price returns to the 1w pivot level or opposite Camarilla level
        pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
        exit_long = position == 1 and (price_close < pivot_1w_aligned[i] or 
                                       price_close < s3_1w_aligned[i])
        exit_short = position == -1 and (price_close > pivot_1w_aligned[i] or 
                                         price_close > r3_1w_aligned[i])
        
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