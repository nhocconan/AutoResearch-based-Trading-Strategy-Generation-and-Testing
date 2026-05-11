#!/usr/bin/env python3
"""
4h_1d_Camarilla_R3_S3_Breakout_Volume_Trend
Hypothesis: Use 1-day Camarilla R3/S3 levels as entry triggers and 4-hour EMA50 for trend filter.
Enter long when price breaks above R3 in 4h uptrend with volume confirmation.
Enter short when price breaks below S3 in 4h downtrend with volume confirmation.
Exit when price returns to the Camarilla center (P) level.
Camarilla levels provide high-probability reversal/intraday turning points.
Trend filter avoids counter-trend trades. Volume confirmation ensures conviction.
Designed for 15-25 trades/year (60-100 total over 4 years) to avoid fee drag.
Works in bull by buying pullbacks in uptrend; works in bear by selling rallies in downtrend.
"""

name = "4h_1d_Camarilla_R3_S3_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1-day data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- 1-day Camarilla Levels (from previous day) ---
    # Calculate from previous day's OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Camarilla formulas
    R3 = close_1d + (high_1d - low_1d) * 1.1 / 2
    S3 = close_1d - (high_1d - low_1d) * 1.1 / 2
    P = (high_1d + low_1d + close_1d) / 3  # Pivot point
    
    # Shift by 1 to use previous day's levels (no look-ahead)
    R3 = np.roll(R3, 1)
    S3 = np.roll(S3, 1)
    P = np.roll(P, 1)
    R3[0] = np.nan
    S3[0] = np.nan
    P[0] = np.nan
    
    # Align 1-day Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    P_aligned = align_htf_to_ltf(prices, df_1d, P)
    
    # --- 4h EMA50 for trend filter ---
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume confirmation: volume > 1.5x 20-period average ---
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(P_aligned[i]) or np.isnan(ema_50[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend based on EMA50
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume confirmation
            if uptrend and close[i] > R3_aligned[i] and volume_ok[i]:
                # Long: 4h uptrend + price breaks above R3 + volume confirmation
                signals[i] = 0.25
                position = 1
            elif downtrend and close[i] < S3_aligned[i] and volume_ok[i]:
                # Short: 4h downtrend + price breaks below S3 + volume confirmation
                signals[i] = -0.25
                position = -1
        else:
            # Exit when price returns to Camarilla pivot level (P)
            if position == 1:
                # Exit long: price returns to or below pivot level
                if close[i] <= P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to or above pivot level
                if close[i] >= P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals