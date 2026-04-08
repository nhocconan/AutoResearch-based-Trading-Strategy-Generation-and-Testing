#!/usr/bin/env python3
# 12h_1d_camarilla_volume_trend_v1
# Hypothesis: 12-hour Camarilla pivot reversal with 1-day volume confirmation and EMA trend filter.
# Long: price touches Camarilla S3 (strong support) AND volume > 1.5x 20-period average AND price > 1-day EMA50.
# Short: price touches Camarilla R3 (strong resistance) AND volume > 1.5x 20-period average AND price < 1-day EMA50.
# Exit: price reaches Camarilla S1/R1 (intraday support/resistance) or opposite Camarilla touch with volume.
# Designed to capture reversals at key institutional levels with volume confirmation to avoid false breaks.
# Works in both bull and bear markets by fading extremes at proven pivot levels.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_camarilla_volume_trend_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12-hour Camarilla pivot levels (based on previous day)
    # Calculate from previous day's OHLC (we'll use daily data shifted by 1)
    camarilla_s3 = np.full(n, np.nan)  # Strong support
    camarilla_s1 = np.full(n, np.nan)  # Intraday support
    camarilla_r1 = np.full(n, np.nan)  # Intraday resistance
    camarilla_r3 = np.full(n, np.nan)  # Strong resistance
    
    # Get daily data for pivot calculation (previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla formulas:
    # Range = high - low
    # S3 = close - 1.1 * (high - low) / 2
    # S1 = close - 1.1 * (high - low) / 4
    # R1 = close + 1.1 * (high - low) / 4
    # R3 = close + 1.1 * (high - low) / 2
    camarilla_s3_1d = close_1d - 1.1 * (high_1d - low_1d) / 2
    camarilla_s1_1d = close_1d - 1.1 * (high_1d - low_1d) / 4
    camarilla_r1_1d = close_1d + 1.1 * (high_1d - low_1d) / 4
    camarilla_r3_1d = close_1d + 1.1 * (high_1d - low_1d) / 2
    
    # Align daily Camarilla levels to 12h timeframe
    camarilla_s3 = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    camarilla_s1 = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    camarilla_r1 = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_r3 = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    
    # 20-period average volume for confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(20, n):
        avg_volume[i] = np.mean(volume[i-20:i])
    
    # 1-day EMA50 for trend filter
    ema_1d_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d_50[i] = close_1d[i] * (2/51) + ema_1d_50[i-1] * (49/51)
    
    ema_1d_50_aligned = align_htf_to_ltf(prices, df_1d, ema_1d_50)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        s3 = camarilla_s3[i]
        s1 = camarilla_s1[i]
        r1 = camarilla_r1[i]
        r3 = camarilla_r3[i]
        ema_1d = ema_1d_50_aligned[i]
        
        if np.isnan(s3) or np.isnan(s1) or np.isnan(r1) or np.isnan(r3) or np.isnan(avg_vol) or np.isnan(ema_1d):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        vol_surge = vol > 1.5 * avg_vol
        
        if position == 1:  # Long position
            if price > r1 or (price > r3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price < s1 or (price < s3 and vol_surge):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price < s3 and vol_surge and price < ema_1d:
                position = 1
                signals[i] = 0.25
            elif price > r3 and vol_surge and price > ema_1d:
                position = -1
                signals[i] = -0.25
    
    return signals