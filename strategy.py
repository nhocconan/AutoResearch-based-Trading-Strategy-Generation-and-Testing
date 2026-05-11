#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_Pivot_Breakout_Volume_StrongTrend
Hypothesis: Price breaks above/below R3/S3 from daily pivots with volume > 20-period average,
strong weekly trend (EMA200), and daily trend alignment. Exits when price returns to opposite side (S1/R1).
Uses 12h timeframe to reduce trade frequency and avoid fee drag. Targets 50-150 total trades over 4 years.
Works in bull/bear via strong trend filter and mean-reversion exit.
"""

name = "12h_1d_1w_Camarilla_Pivot_Breakout_Volume_StrongTrend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Get 1d and 1w data for pivot calculation and trend filters
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1w Strong Trend Filter: EMA200 ---
    close_1w = df_1w['close'].values
    ema200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # --- Camarilla Pivots from 1d (previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's data
    camarilla_high = np.full_like(close_1d, np.nan)
    camarilla_low = np.full_like(close_1d, np.nan)
    camarilla_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's OHLC to calculate today's pivots
        camarilla_high[i] = high_1d[i-1]
        camarilla_low[i] = low_1d[i-1]
        camarilla_close[i] = close_1d[i-1]
    
    # Calculate Camarilla levels
    R4 = camarilla_close + ((camarilla_high - camarilla_low) * 1.5000)
    R3 = camarilla_close + ((camarilla_high - camarilla_low) * 1.2500)
    R2 = camarilla_close + ((camarilla_high - camarilla_low) * 1.1666)
    R1 = camarilla_close + ((camarilla_high - camarilla_low) * 1.0833)
    PP = camarilla_close
    S1 = camarilla_close - ((camarilla_high - camarilla_low) * 1.0833)
    S2 = camarilla_close - ((camarilla_high - camarilla_low) * 1.1666)
    S3 = camarilla_close - ((camarilla_high - camarilla_low) * 1.2500)
    S4 = camarilla_close - ((camarilla_high - camarilla_low) * 1.5000)
    
    # Align pivots to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    R1_12h = align_htf_to_ltf(prices, df_1d, R1)
    S1_12h = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 200  # for EMA200 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1w_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trends
        daily_trend_up = close[i] > ema50_1d_aligned[i]
        daily_trend_down = close[i] < ema50_1d_aligned[i]
        weekly_trend_up = close[i] > ema200_1w_aligned[i]
        weekly_trend_down = close[i] < ema200_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of both daily and weekly trend with volume
            if close[i] > R3_12h[i] and daily_trend_up and weekly_trend_up and vol_ok:
                # Long: price breaks above R3 + daily/weekly uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close[i] < S3_12h[i] and daily_trend_down and weekly_trend_down and vol_ok:
                # Short: price breaks below S3 + daily/weekly downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close[i] <= S1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close[i] >= R1_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals