#!/usr/bin/env python3
"""
12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume
Hypothesis: Camarilla R3/S3 breakout on 12h with 1d trend filter and volume confirmation.
- Long when: price breaks above R3, 1d EMA34 uptrend, volume > 20-period average
- Short when: price breaks below S3, 1d EMA34 downtrend, volume > 20-period average
- Exit when price returns to H4/L4 or trend reverses
Camarilla levels provide institutional support/resistance. Trend filter ensures alignment.
Volume confirms breakout strength. Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in bull by riding breakouts, in bear by catching breakdowns with trend alignment.
"""

name = "12h_1d_Camarilla_R3_S3_Breakout_Trend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from previous 1d (calculated on 12h close) ---
    # Use previous day's OHLC to calculate today's Camarilla levels
    high_prev = df_1d['high'].values
    low_prev = df_1d['low'].values
    close_prev = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    R3 = close_prev + 1.1 * (high_prev - low_prev) / 6
    S3 = close_prev - 1.1 * (high_prev - low_prev) / 6
    H4 = close_prev + 1.1 * (high_prev - low_prev) / 2
    L4 = close_prev - 1.1 * (high_prev - low_prev) / 2
    
    # Align Camarilla levels to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40  # for EMA34 calculation
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(H4_aligned[i]) or 
            np.isnan(L4_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema34_1d_aligned[i]
        trend_down = close_12h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakout entries only in direction of 1d trend with volume
            if close_12h[i] > R3_aligned[i] and trend_up and vol_ok:
                # Long: price breaks above R3 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_12h[i] < S3_aligned[i] and trend_down and vol_ok:
                # Short: price breaks below S3 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to H4 OR trend turns down
                if close_12h[i] <= H4_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to L4 OR trend turns up
                if close_12h[i] >= L4_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals