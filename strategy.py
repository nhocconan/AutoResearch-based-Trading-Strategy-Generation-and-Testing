#!/usr/bin/env python3
# 1h_4h1d_Camarilla_Trend_Filter
# Hypothesis: 1h breakout at 4h/1d Camarilla R3/S3 with 4h trend filter and volume spike.
# Uses 4h trend (close > EMA50) for bias, reducing counter-trend trades.
# Volume surge (1.5x 24-period MA) confirms institutional participation.
# Designed for 1h timeframe to target 15-37 trades/year per symbol.
# Works in bull/bear by requiring trend alignment, avoiding chop whipsaws.

name = "1h_4h1d_Camarilla_Trend_Filter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 4h data for trend and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # 1h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h Camarilla levels (using previous bar's OHLC)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    camarilla_r3 = np.full(len(df_4h), np.nan)
    camarilla_s3 = np.full(len(df_4h), np.nan)
    
    for i in range(1, len(df_4h)):
        prev_high = high_4h[i-1]
        prev_low = low_4h[i-1]
        prev_close = close_4h[i-1]
        diff = prev_high - prev_low
        
        camarilla_r3[i] = prev_close + diff * 1.1 / 4
        camarilla_s3[i] = prev_close - diff * 1.1 / 4
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3)
    
    # Volume average (24-period for 1h = 24 hours)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for 4h EMA50 + vol MA
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_4h_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 4h close > EMA50
        close_4h_aligned = align_htf_to_ltf(prices, df_4h, df_4h['close'].values)
        uptrend = close_4h_aligned[i] > ema_50_4h_aligned[i]
        downtrend = close_4h_aligned[i] < ema_50_4h_aligned[i]
        
        # Volume confirmation (1.5x average for significance)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Camarilla R3 in uptrend with volume spike
            if close[i] > camarilla_r3_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.20
                position = 1
            # Short: Breakdown below Camarilla S3 in downtrend with volume spike
            elif close[i] < camarilla_s3_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.20
                position = -1
        else:
            if position == 1:
                # Long exit: close below Camarilla R3 or trend fails
                if close[i] < camarilla_r3_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: close above Camarilla S3 or trend fails
                if close[i] > camarilla_s3_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals