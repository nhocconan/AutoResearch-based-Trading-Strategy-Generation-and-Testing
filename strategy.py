#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Volume
Hypothesis: 12h timeframe with 1d Camarilla pivot breakouts, volume confirmation, and 1d trend filter.
Designed for lower trade frequency (12-37/year) to minimize fee drag while capturing sustained trends.
Works in bull markets via breakout momentum and in bear markets via trend-filtered mean reversion to opposite pivot.
"""

name = "12h_1d_Camarilla_Pivot_Breakout_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get 1d data for pivot calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- Camarilla Pivots from 1d (previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivots from previous day's OHLC
    camarilla_high = np.full_like(close_1d, np.nan)
    camarilla_low = np.full_like(close_1d, np.nan)
    camarilla_close = np.full_like(close_1d, np.nan)
    
    for i in range(1, len(close_1d)):
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
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 60  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or 
            np.isnan(R1_12h[i]) or np.isnan(S1_12h[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_12h[i] > R3_12h[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < S3_12h[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume
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