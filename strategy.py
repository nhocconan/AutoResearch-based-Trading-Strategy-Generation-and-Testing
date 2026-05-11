#!/usr/bin/env python3
"""
6h_1w_1d_Camarilla_R3S3_Breakout_WeeklyTrend
Hypothesis: On 6B timeframe, trade breakouts of daily Camarilla R3/S3 levels only when aligned with weekly trend (EMA50).
Weekly trend filter reduces false breakouts in ranging markets and improves win rate.
Entry: Long when price closes above R3 + weekly EMA50 uptrend; Short when price closes below S3 + weekly EMA50 downtrend.
Exit: When price returns to opposite S1/R1 level or weekly trend flips.
Position size: 0.25. Targets 15-30 trades/year (60-120 over 4 years) to minimize fee drag.
Works in bull/bear: Weekly EMA50 adapts to trend direction, breakouts capture momentum in trending regimes.
"""

name = "6h_1w_1d_Camarilla_R3S3_Breakout_WeeklyTrend"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get weekly data for trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Weekly Trend Filter: EMA50 ---
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Daily Camarilla Pivots (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Use previous day's OHLC to calculate today's Camarilla levels
    phigh = np.roll(high_1d, 1)
    plow = np.roll(low_1d, 1)
    pclose = np.roll(close_1d, 1)
    phigh[0] = np.nan
    plow[0] = np.nan
    pclose[0] = np.nan
    
    # Calculate Camarilla levels
    R3 = pclose + ((phigh - plow) * 1.2500)
    S3 = pclose - ((phigh - plow) * 1.2500)
    R1 = pclose + ((phigh - plow) * 1.0833)
    S1 = pclose - ((phigh - plow) * 1.0833)
    
    # Align to 6h timeframe
    R3_6h = align_htf_to_ltf(prices, df_1d, R3)
    S3_6h = align_htf_to_ltf(prices, df_1d, S3)
    R1_6h = align_htf_to_ltf(prices, df_1d, R1)
    S1_6h = align_htf_to_ltf(prices, df_1d, S1)
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for weekly EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(R3_6h[i]) or np.isnan(S3_6h[i]) or 
            np.isnan(R1_6h[i]) or np.isnan(S1_6h[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_up = close_6h[i] > ema50_1w_aligned[i]
        weekly_down = close_6h[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if close_6h[i] > R3_6h[i] and weekly_up and vol_ok:
                # Long: price breaks above R3 + weekly uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_6h[i] < S3_6h[i] and weekly_down and vol_ok:
                # Short: price breaks below S3 + weekly downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to S1 or weekly trend flips down
                if close_6h[i] <= S1_6h[i] or not weekly_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to R1 or weekly trend flips up
                if close_6h[i] >= R1_6h[i] or not weekly_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals