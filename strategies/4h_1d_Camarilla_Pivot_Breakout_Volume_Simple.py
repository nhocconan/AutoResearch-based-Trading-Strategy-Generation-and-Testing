#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Simple
Hypothesis: Camarilla pivot levels from 1d combined with volume confirmation and 1d trend filter.
- Long when: price breaks above R3 with volume > 20-period average and 1d EMA50 uptrend
- Short when: price breaks below S3 with volume > 20-period average and 1d EMA50 downtrend
- Exit when price returns to opposite pivot level (S1 for longs, R1 for shorts)
Targets 20-40 trades/year (80-160 over 4 years) to minimize fee drift.
Uses 1d trend filter to avoid counter-trend trades in ranging markets.
"""

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Simple"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
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
    
    # Align pivots to 4h timeframe
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_4h[i]) or np.isnan(S3_4h[i]) or 
            np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_4h[i] > R3_4h[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < S3_4h[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 (opposite side)
                if close_4h[i] <= S1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 (opposite side)
                if close_4h[i] >= R1_4h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals