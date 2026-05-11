#!/usr/bin/env python3
"""
1d_Camarilla_R3S3_Breakout_1wTrend_VolumeFilter
Hypothesis: Breakout above weekly R3 or below weekly S3 with volume confirmation and 1d trend filter.
- Weekly timeframe provides stronger trend filter than daily, reducing false breakouts in chop.
- Volume confirmation ensures breakout conviction.
- Target 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
- Works in bull (catch breakouts) and bear (avoid counter-trend via weekly trend).
"""

name = "1d_Camarilla_R3S3_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter and pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- Weekly Trend Filter: EMA20 ---
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # --- Weekly Camarilla Pivots (from previous week) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots from previous week's OHLC
    camarilla_high = np.full_like(close_1w, np.nan)
    camarilla_low = np.full_like(close_1w, np.nan)
    camarilla_close = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        # Use previous week's OHLC to calculate current week's pivots
        camarilla_high[i] = high_1w[i-1]
        camarilla_low[i] = low_1w[i-1]
        camarilla_close[i] = close_1w[i-1]
    
    # Calculate Camarilla levels
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    
    # Align pivots to 1d timeframe
    R3_1d = align_htf_to_ltf(prices, df_1w, R3)
    S3_1d = align_htf_to_ltf(prices, df_1w, S3)
    R1_1d = align_htf_to_ltf(prices, df_1w, R1)
    S1_1d = align_htf_to_ltf(prices, df_1w, S1)
    
    # --- Volume Confirmation: 1d volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA20 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_1d[i]) or np.isnan(S3_1d[i]) or 
            np.isnan(R1_1d[i]) or np.isnan(S1_1d[i]) or
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close_1d[i] > ema20_1w_aligned[i]
        trend_down = close_1d[i] < ema20_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_1d[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if close_1d[i] > R3_1d[i] and trend_up and vol_ok:
                # Long: price breaks above weekly R3 + weekly uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_1d[i] < S3_1d[i] and trend_down and vol_ok:
                # Short: price breaks below weekly S3 + weekly downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close_1d[i] <= S1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close_1d[i] >= R1_1d[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals