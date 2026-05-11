#!/usr/bin/env python3
"""
12h_1w_Camarilla_Pivot_Breakout_Trend_Volume
Hypothesis: Weekly Camarilla pivot breakouts with weekly trend filter and volume confirmation.
- Long when price breaks above weekly R3, weekly trend is up, and volume > 20-period average
- Short when price breaks below weekly S3, weekly trend is down, and volume > 20-period average
- Exit when price returns to opposite weekly pivot level (S1 for longs, R1 for shorts)
- Weekly timeframe reduces trade frequency to target 12-37 trades/year
- Works in both bull and bear markets via trend filter that avoids counter-trend trades
"""

name = "12h_1w_Camarilla_Pivot_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for pivot calculation and trend filter (called ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- Weekly Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Weekly Camarilla Pivots (from previous week) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_vals = df_1w['close'].values
    
    # Calculate pivots from previous week's data
    camarilla_high = np.full_like(close_1w_vals, np.nan)
    camarilla_low = np.full_like(close_1w_vals, np.nan)
    camarilla_close = np.full_like(close_1w_vals, np.nan)
    
    for i in range(1, len(close_1w_vals)):
        # Use previous week's OHLC to calculate this week's pivots
        camarilla_high[i] = high_1w[i-1]
        camarilla_low[i] = low_1w[i-1]
        camarilla_close[i] = close_1w_vals[i-1]
    
    # Calculate Camarilla levels
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    
    # Align pivots to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1w, R3)
    S3_12h = align_htf_to_ltf(prices, df_1w, S3)
    R1_12h = align_htf_to_ltf(prices, df_1w, R1)
    S1_12h = align_htf_to_ltf(prices, df_1w, S1)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_12h[i] > ema50_1w_aligned[i]
        trend_down = close_12h[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if close_12h[i] > R3_12h[i] and trend_up and vol_ok:
                # Long: price breaks above weekly R3 + weekly uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < S3_12h[i] and trend_down and vol_ok:
                # Short: price breaks below weekly S3 + weekly downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close_12h[i] <= S1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close_12h[i] >= R1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals