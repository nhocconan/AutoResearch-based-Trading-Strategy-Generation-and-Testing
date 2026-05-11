#!/usr/bin/env python3
"""
4h_1d_Camarilla_Pivot_Breakout_Volume_Trend
Hypothesis: Price breaking above/below daily Camarilla R3/S3 with volume confirmation and 1d EMA50 trend filter reduces false breakouts, yielding fewer but higher-quality trades. Exit at opposite S1/R1 levels. Designed to work in both bull and bear markets via trend alignment.
"""

name = "4h_1d_Camarilla_Pivot_Breakout_Volume_Trend"
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
    
    # --- 1d ATR for volatility filter ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # --- Camarilla Pivots from 1d (previous day) ---
    # Use previous day's OHLC to calculate today's pivots
    # Shift arrays by 1 to get previous day's values
    phigh_1d = np.concatenate([[np.nan], high_1d[:-1]])
    plow_1d = np.concatenate([[np.nan], low_1d[:-1]])
    pclose_1d = np.concatenate([[np.nan], close_1d[:-1]])
    
    # Calculate Camarilla levels from previous day
    range_1d = phigh_1d - plow_1d
    R3 = pclose_1d + (range_1d * 1.2500)
    S3 = pclose_1d - (range_1d * 1.2500)
    R1 = pclose_1d + (range_1d * 1.0833)
    S1 = pclose_1d - (range_1d * 1.0833)
    
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
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(atr_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: avoid low volatility periods
        vol_filter = atr_1d_aligned[i] > np.nanpercentile(atr_1d_aligned[:i+1], 30)
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema50_1d_aligned[i]
        trend_down = close_4h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume and volatility
            if close_4h[i] > R3_4h[i] and trend_up and vol_ok and vol_filter:
                # Long: price breaks above R3 + 1d uptrend + volume + volatility
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < S3_4h[i] and trend_down and vol_ok and vol_filter:
                # Short: price breaks below S3 + 1d downtrend + volume + volatility
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