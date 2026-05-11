#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend
Hypothesis: On 12h chart, price breaking Camarilla R3/S3 levels with weekly trend filter (1w EMA50) and volume confirmation captures institutional breakout moves. Works in bull (buy R3 breakouts in uptrend) and bear (sell S3 breakdowns in downtrend). Weekly trend filter avoids counter-trend trades. Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1wTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly EMA50 trend ---
    close_1w = df_1w['close'].values
    ema_1w = np.full(len(close_1w), np.nan)
    for i in range(len(close_1w)):
        if i < 50:
            ema_1w[i] = np.nan
        elif i == 50:
            ema_1w[i] = np.mean(close_1w[0:50])
        else:
            ema_1w[i] = (close_1w[i] * 2 / (50 + 1)) + (ema_1w[i-1] * (49 / (50 + 1)))
    
    # --- Daily Camarilla levels (using previous day's OHLC) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3 and S3
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        rang = ph - pl
        camarilla_r3[i] = pc + (rang * 1.1 / 4)
        camarilla_s3[i] = pc - (rang * 1.1 / 4)
    
    # --- 12h volume MA(20) ---
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Align weekly EMA50 to 12h
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # Align daily Camarilla levels to 12h
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max(weekly EMA50 needs 50, daily Camarilla needs 1, vol MA20)
    start_idx = max(50, 1, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_1w_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or
            np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > vol_ma[i] * 1.5
        
        if position == 0:
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_1w_aligned[i] and vol_spike:
                # Long: break above R3 in weekly uptrend
                signals[i] = 0.25
                position = 1
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_1w_aligned[i] and vol_spike:
                # Short: break below S3 in weekly downtrend
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Exit long: price closes below weekly EMA50 OR re-enters Camarilla body
                if close[i] < ema_1w_aligned[i] or (camarilla_s3_aligned[i] < close[i] < camarilla_r3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price closes above weekly EMA50 OR re-enters Camarilla body
                if close[i] > ema_1w_aligned[i] or (camarilla_s3_aligned[i] < close[i] < camarilla_r3_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals