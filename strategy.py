#!/usr/bin/env python3
# 1h_4h1d_Camarilla_Breakout_Trend_Volume
# Hypothesis: Buy breakouts above Camarilla R1 in uptrends (4h EMA50 > 200 EMA + 1d close > EMA50) and sell breakdowns below S3 in downtrends, with volume confirmation.
# Uses 4h/1d for trend direction and structure, 1h for precise entry timing. Targets 60-150 trades over 4 years via strict multi-condition entry.
# Works in bull/bear by requiring trend alignment, avoiding counter-trend traps.

name = "1h_4h1d_Camarilla_Breakout_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 4h and 1d data for trend and structure
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels from previous day
    # R1 = C + (H-L)*1.1/12, S3 = C - (H-L)*1.1/4
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    camarilla_r1 = prev_close + (prev_high - prev_low) * 1.1 / 12
    camarilla_s3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # 4h EMA50 and EMA200 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200_4h = pd.Series(df_4h['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    ema_200_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_200_4h)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume average (24-period for 1h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for Camarilla (previous day) + EMAs + vol MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_aligned[i]) or
            np.isnan(s3_aligned[i]) or
            np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(ema_200_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 4h EMA50 > EMA200 AND 1h close > 4h EMA50 AND 1d close > 1d EMA50
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h['close'].values)
        uptrend_4h = close_4h_aligned[i] > ema_50_4h_aligned[i] and ema_50_4h_aligned[i] > ema_200_4h_aligned[i]
        uptrend_1d = close_4h_aligned[i] > ema_50_1d_aligned[i]  # Approximate 1d close using 4h close aligned
        # Better: get actual 1d close aligned
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend_1d = close_1d_aligned[i] > ema_50_1d_aligned[i]
        uptrend = uptrend_4h and uptrend_1d
        
        # Downtrend: 4h EMA50 < EMA200 AND 1h close < 4h EMA50 AND 1d close < 1d EMA50
        downtrend_4h = close_4h_aligned[i] < ema_50_4h_aligned[i] and ema_50_4h_aligned[i] < ema_200_4h_aligned[i]
        downtrend_1d = close_1d_aligned[i] < ema_50_1d_aligned[i]
        downtrend = downtrend_4h and downtrend_1d
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above R1 in uptrend with volume
            if close[i] > r1_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below S3 in downtrend with volume
            elif close[i] < s3_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: close below R1 or trend fails
                if close[i] < r1_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: close above S3 or trend fails
                if close[i] > s3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals