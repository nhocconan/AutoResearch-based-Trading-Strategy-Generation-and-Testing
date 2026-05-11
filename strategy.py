#!/usr/bin/env python3
"""
12h_1w_Camarilla_R3S3_Breakout_Trend_1dVol
Hypothesis: Use weekly Camarilla R3/S3 levels as key institutional support/resistance.
Breakout above R3 with 1w uptrend and daily volume surge = long.
Breakdown below S3 with 1w downtrend and daily volume surge = short.
Camarilla levels work well in ranging markets (2025-2026) while trend filter captures breaks.
Weekly timeframe reduces noise, daily volume confirms institutional participation.
Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
"""

name = "12h_1w_Camarilla_R3S3_Breakout_Trend_1dVol"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for trend and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly Camarilla Levels (R3, S3) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla for each weekly bar
    R3 = np.full_like(close_1w, np.nan)
    S3 = np.full_like(close_1w, np.nan)
    for i in range(len(close_1w)):
        if i == 0 or np.isnan(high_1w[i]) or np.isnan(low_1w[i]) or np.isnan(close_1w[i-1]):
            continue
        range_ = high_1w[i] - low_1w[i]
        C = close_1w[i-1]  # previous close
        R3[i] = C + (range_ * 1.1 / 6)
        S3[i] = C - (range_ * 1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1w, R3)
    S3_12h = align_htf_to_ltf(prices, df_1w, S3)
    
    # --- Daily Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Daily Volume Confirmation: 20-period average ---
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current daily volume > 20-day average
        vol_ok = volume_12h[i] > vol_ma_20_aligned[i]
        
        if position == 0:
            # Look for breakouts in direction of daily trend with volume
            if close_12h[i] > R3_12h[i] and close_12h[i] > ema50_1d_aligned[i] and vol_ok:
                # Long: break above R3 + daily uptrend + volume surge
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < S3_12h[i] and close_12h[i] < ema50_1d_aligned[i] and vol_ok:
                # Short: break below S3 + daily downtrend + volume surge
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price returns to Camarilla core (between R3 and S3) or trend reverses
            if position == 1:
                # Exit long: price back below R3 OR trend turns down
                if close_12h[i] < R3_12h[i] or close_12h[i] < ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price back above S3 OR trend turns up
                if close_12h[i] > S3_12h[i] or close_12h[i] > ema50_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals