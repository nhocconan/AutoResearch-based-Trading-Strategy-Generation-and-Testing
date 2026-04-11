#!/usr/bin/env python3
# 4h_1d_camarilla_breakout_volume_trend_v4
# Strategy: 4-hour Camarilla pivot breakout with 1-day trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Uses daily Camarilla pivot levels (R4/S4 for breakouts, R3/S3 for rejections)
# filtered by 1-day EMA50 trend and volume spikes. Designed to capture multi-day momentum
# bursts while filtering counter-trend noise. Tightened entry conditions to reduce trades
# and avoid overtrading. Focus on BTC/ETH robustness.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_camarilla_breakout_volume_trend_v4"
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
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d OHLC for Camarilla pivots and EMA50
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R4 = C + (H-L) * 1.1/2
    # R3 = C + (H-L) * 1.1/4
    # S3 = C - (H-L) * 1.1/4
    # S4 = C - (H-L) * 1.1/2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r4_1d = close_1d + range_1d * 1.1 / 2.0
    r3_1d = close_1d + range_1d * 1.1 / 4.0
    s3_1d = close_1d - range_1d * 1.1 / 4.0
    s4_1d = close_1d - range_1d * 1.1 / 2.0
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d data to 4h timeframe (wait for daily close)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_avg)  # Increased threshold for stricter volume confirmation
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s4_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        price_close = close[i]
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = price_close > ema_50_1d_aligned[i]
        downtrend_1d = price_close < ema_50_1d_aligned[i]
        
        # Camarilla breakout signals (using previous day's levels)
        breakout_up = price_close > r4_1d_aligned[i]   # Break above R4
        breakdown_down = price_close < s4_1d_aligned[i]  # Break below S4
        reject_at_r3 = price_close < r3_1d_aligned[i] and price_close > r3_1d_aligned[i-1]  # Reject at R3
        reject_at_s3 = price_close > s3_1d_aligned[i] and price_close < s3_1d_aligned[i-1]  # Reject at S3
        
        # Volume confirmation
        vol_confirmed = vol_spike[i]
        
        # Long: Break above R4 with volume in uptrend OR rejection at R3 with volume in uptrend
        long_signal = (breakout_up and vol_confirmed and uptrend_1d) or \
                      (reject_at_r3 and vol_confirmed and uptrend_1d)
        
        # Short: Break below S4 with volume in downtrend OR rejection at S3 with volume in downtrend
        short_signal = (breakdown_down and vol_confirmed and downtrend_1d) or \
                       (reject_at_s3 and vol_confirmed and downtrend_1d)
        
        # Exit when price returns to the 1d pivot level or opposite Camarilla level
        exit_long = position == 1 and (price_close < pivot_1d_aligned[i] or 
                                       price_close < s3_1d_aligned[i])
        exit_short = position == -1 and (price_close > pivot_1d_aligned[i] or 
                                         price_close > r3_1d_aligned[i])
        
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