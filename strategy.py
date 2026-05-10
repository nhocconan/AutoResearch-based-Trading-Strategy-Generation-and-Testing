#!/usr/bin/env python3
# 6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume
# Hypothesis: 6h breakout at daily Camarilla R3/S3 levels with daily trend filter and volume confirmation.
# Uses daily trend (close > EMA50) to avoid counter-trend trades. Volume surge (2x 20-period MA) confirms institutional participation.
# Designed for 6h timeframe targeting 12-37 trades/year per symbol. Works in bull/bear by requiring trend alignment and volume confirmation to reduce whipsaws.

name = "6h_12h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for trend and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate daily Camarilla levels (using previous day's OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(df_1d), np.nan)
    camarilla_s3 = np.full(len(df_1d), np.nan)
    
    for i in range(1, len(df_1d)):
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        diff = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + diff * 1.1 / 4
        camarilla_s3[i] = prev_close - diff * 1.1 / 4
    
    # Align daily indicators to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Volume average (20-period for 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for daily EMA50 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: daily close > EMA50
        close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
        uptrend = close_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend = close_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (2x average for significance)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Camarilla R3 in uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Camarilla S3 in downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Camarilla R3 or trend fails
                if close[i] < camarilla_r3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Camarilla S3 or trend fails
                if close[i] > camarilla_s3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals