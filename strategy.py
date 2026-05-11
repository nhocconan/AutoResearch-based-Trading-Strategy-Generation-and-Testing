#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R3S3_Breakout_WeeklyTrend
Hypothesis: Breakout of weekly Camarilla R3/S3 levels with 1d trend filter and volume confirmation.
- Long when: price breaks above weekly R3 with volume > 20-period average and 1d EMA50 uptrend
- Short when: price breaks below weekly S3 with volume > 20-period average and 1d EMA50 downtrend
- Exit when price returns to opposite weekly level (S1 for longs, R1 for shorts)
- Uses weekly structure for direction, 1d for trend filter to avoid counter-trend trades
- Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
"""

name = "6h_1w_1d_Camarilla_R3S3_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for Camarilla pivots and daily data for trend filter
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1w) < 2 or len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Weekly Camarilla Pivots (previous week) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots from previous week's data
    camarilla_high = np.full_like(close_1w, np.nan)
    camarilla_low = np.full_like(close_1w, np.nan)
    camarilla_close = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        # Use previous week's OHLC to calculate this week's pivots
        camarilla_high[i] = high_1w[i-1]
        camarilla_low[i] = low_1w[i-1]
        camarilla_close[i] = close_1w[i-1]
    
    # Calculate weekly Camarilla levels
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    
    # Align weekly pivots to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1w, R3)
    S3_6h = align_htf_to_ltf(prices, df_1w, S3)
    R1_6h = align_htf_to_ltf(prices, df_1w, R1)
    S1_6h = align_htf_to_ltf(prices, df_1w, S1)
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_6h[i] > ema50_1d_aligned[i]
        trend_down = close_6h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_6h[i] > R3_6h[i] and trend_up and vol_ok:
                # Long: price breaks above weekly R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_6h[i] < S3_6h[i] and trend_down and vol_ok:
                # Short: price breaks below weekly S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to weekly S1 (opposite side)
                if close_6h[i] <= S1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly R1 (opposite side)
                if close_6h[i] >= R1_6h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals