#!/usr/bin/env python3
# 4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
# Hypothesis: Use 1d Camarilla R3/S3 levels as breakout triggers, filtered by 1d EMA34 trend and volume spike.
# Works in bull markets by buying R3 breakouts with uptrend, and in bear markets by selling S3 breakdowns with downtrend.
# Volume confirmation ensures breakouts have strength. Designed for moderate trade frequency (~25-40/year).

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Camarilla levels (R3, S3) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            # Camarilla formula: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
            camarilla_r3[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
            camarilla_s3[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) / 2
    
    # --- 1d EMA34 trend ---
    ema_1d = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if i < 34:
            ema_1d[i] = np.nan
        elif i == 34:
            ema_1d[i] = np.mean(close_1d[0:34])
        else:
            ema_1d[i] = (close_1d[i] * 2 / (34 + 1)) + (ema_1d[i-1] * (32 / (34 + 1)))
    
    # EMA slope (rising/falling)
    ema_slope = np.full(len(close_1d), np.nan)
    for i in range(35, len(close_1d)):
        ema_slope[i] = ema_1d[i] - ema_1d[i-1]
    
    # --- Volume confirmation (volume > 20-period average) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align 1d indicators to 4h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: enough for Camarilla (need prev day), EMA34, and volume MA(20)
    start_idx = max(1, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(ema_slope_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Camarilla breakout conditions
        breakout_r3 = close[i] > camarilla_r3_aligned[i]
        breakdown_s3 = close[i] < camarilla_s3_aligned[i]
        
        # Volume spike condition
        vol_spike = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        if position == 0:
            if breakout_r3 and ema_slope_aligned[i] > 0 and vol_spike:
                # Long: R3 breakout + rising EMA34 + volume spike
                signals[i] = 0.25
                position = 1
            elif breakdown_s3 and ema_slope_aligned[i] < 0 and vol_spike:
                # Short: S3 breakdown + falling EMA34 + volume spike
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price falls back to S3 level OR EMA34 slope turns negative
                if close[i] < camarilla_s3_aligned[i] or ema_slope_aligned[i] < 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price rises back to R3 level OR EMA34 slope turns positive
                if close[i] > camarilla_r3_aligned[i] or ema_slope_aligned[i] > 0:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals