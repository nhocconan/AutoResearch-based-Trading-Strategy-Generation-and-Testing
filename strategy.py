#!/usr/bin/env python3
# 4h_Pivot_Reversal_with_Volume_and_Trend
# Hypothesis: Reversals at daily Camarilla pivot levels (S3/R3) with volume confirmation and 1d trend filter.
# Long when price crosses above S3 in 1d uptrend with volume surge; short when crosses below R3 in 1d downtrend.
# Works in bull markets (buy dips at support) and bear markets (sell rallies at resistance) by fading extremes.
# Uses volume to avoid false breaks and pivot levels as institutional reference points.

name = "4h_Pivot_Reversal_with_Volume_and_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_ltf_to_hlf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Daily Camarilla levels (S3, R3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Typical price for Camarilla calculation
    typical_price = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    # Camarilla levels: S3 = close - 1.1*(H-L)/2, R3 = close + 1.1*(H-L)/2
    s3 = close_1d - 1.1 * range_1d / 2
    r3 = close_1d + 1.1 * range_1d / 2
    
    # --- Daily EMA34 for trend filter ---
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_slope = ema_34_1d - np.roll(ema_34_1d, 1)
    ema_34_1d_slope[0] = 0
    ema_34_1d_slope = pd.Series(ema_34_1d_slope).ewm(span=3, adjust=False, min_periods=1).mean().values
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_surge = volume > vol_ma
    
    # Align daily levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_34_1d_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for EMA34 (34) and smoothing (3)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(s3_aligned[i]) or
            np.isnan(r3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(ema_34_1d_slope_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend direction from daily EMA34 slope
        uptrend = ema_34_1d_slope_aligned[i] > 0
        downtrend = ema_34_1d_slope_aligned[i] < 0
        
        if position == 0:
            if uptrend and vol_surge[i]:
                # Long: 1d uptrend + volume surge + price crosses above S3
                if close[i] > s3_aligned[i] and close[i-1] <= s3_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            elif downtrend and vol_surge[i]:
                # Short: 1d downtrend + volume surge + price crosses below R3
                if close[i] < r3_aligned[i] and close[i-1] >= r3_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        else:
            if position == 1:
                # Exit long: price crosses below S3 or trend turns down
                if close[i] < s3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above R3 or trend turns up
                if close[i] > r3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals