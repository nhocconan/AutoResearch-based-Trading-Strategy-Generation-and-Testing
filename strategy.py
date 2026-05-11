#!/usr/bin/env python3
"""
1d_1w_Camarilla_R3S3_Breakout_Trend_Volume
Hypothesis: On daily timeframe, breakouts above weekly R3 or below weekly S3 with volume confirmation and weekly trend filter capture trending moves while avoiding range-bound noise. Weekly trend filter ensures alignment with higher timeframe momentum. Volume confirmation filters false breakouts. Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
"""

name = "1d_1w_Camarilla_R3S3_Breakout_Trend_Volume"
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
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly Trend Filter: EMA21 ---
    close_1w = df_1w['close'].values
    ema21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema21_1w)
    
    # --- Weekly Camarilla Pivots from previous week ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivots from previous week's data
    camarilla_high = np.full_like(close_1w, np.nan)
    camarilla_low = np.full_like(close_1w, np.nan)
    camarilla_close = np.full_like(close_1w, np.nan)
    
    for i in range(1, len(close_1w)):
        camarilla_high[i] = high_1w[i-1]
        camarilla_low[i] = low_1w[i-1]
        camarilla_close[i] = close_1w[i-1]
    
    # Calculate Camarilla levels
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    
    # Align pivots to daily timeframe
    R3_daily = align_htf_to_ltf(prices, df_1w, R3)
    S3_daily = align_htf_to_ltf(prices, df_1w, S3)
    R1_daily = align_htf_to_ltf(prices, df_1w, R1)
    S1_daily = align_htf_to_ltf(prices, df_1w, S1)
    
    # --- Volume Confirmation: daily volume > 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA21 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_daily[i]) or np.isnan(S3_daily[i]) or 
            np.isnan(R1_daily[i]) or np.isnan(S1_daily[i]) or
            np.isnan(ema21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        trend_up = close[i] > ema21_1w_aligned[i]
        trend_down = close[i] < ema21_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if close[i] > R3_daily[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + weekly uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close[i] < S3_daily[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + weekly downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close[i] <= S1_daily[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close[i] >= R1_daily[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals