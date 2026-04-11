# 1d_1w_camarilla_breakout_v1
# Hypothesis: Use weekly price channels to identify trend direction and daily Camarilla levels for precise entries.
# In bull markets: long on pullback to daily S3 in uptrend (weekly close > weekly open).
# In bear markets: short on bounce to daily R3 in downtrend (weekly close < weekly open).
# Weekly trend filter reduces false signals; Camarilla levels provide high-probability mean-reversion entries.
# Target: 1-2 trades per month (12-24/year) to minimize fee decay while capturing meaningful moves.
# Works in both bull and bear markets by trading with the weekly trend.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return signals
    
    # Calculate weekly trend: bullish if close > open, bearish if close < open
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_trend = np.where(weekly_close > weekly_open, 1, -1)  # 1=bullish, -1=bearish
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return signals
    
    # Calculate daily Camarilla pivots
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point and levels
    pivot_1d = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Camarilla levels (key levels: S3/R3 for entries, S4/R4 for stops)
    r3_1d = close_1d + range_1d * 1.1 / 4
    s3_1d = close_1d - range_1d * 1.1 / 4
    r4_1d = close_1d + range_1d * 1.1 / 2
    s4_1d = close_1d - range_1d * 1.1 / 2
    
    # Align weekly trend and daily Camarilla to daily timeframe
    weekly_trend_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # Volume confirmation: volume > 1.3x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(weekly_trend_aligned[i]) or np.isnan(r3_1d_aligned[i]) or
            np.isnan(s3_1d_aligned[i]) or np.isnan(r4_1d_aligned[i]) or
            np.isnan(s4_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price_close = close[i]
        volume_current = volume[i]
        weekly_trend_val = weekly_trend_aligned[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        
        # Volume confirmation
        volume_confirmed = volume_current > 1.3 * vol_ma_20[i]
        
        # Trading logic based on weekly trend
        if weekly_trend_val == 1:  # Weekly uptrend - look for longs
            # Enter long on pullback to S3 with volume
            if price_close <= s3 and volume_confirmed and position != 1:
                position = 1
                signals[i] = 0.25
            # Exit long on break below S4 (stop) or at R3 (target)
            elif position == 1 and (price_close < s4 or price_close >= r3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25 if position == 1 else 0.0
                
        else:  # Weekly downtrend - look for shorts
            # Enter short on bounce to R3 with volume
            if price_close >= r3 and volume_confirmed and position != -1:
                position = -1
                signals[i] = -0.25
            # Exit short on break above R4 (stop) or at S3 (target)
            elif position == -1 and (price_close > r4 or price_close <= s3):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25 if position == -1 else 0.0
    
    return signals