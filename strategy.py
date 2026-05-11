#!/usr/bin/env python3
"""
1h_4d_1d_Camarilla_R1_S1_Breakout_TrendVolume
Hypothesis: Use daily Camarilla R1/S1 levels as breakout triggers with 4h trend filter (4h EMA50) and volume confirmation.
In 4h uptrend (price above 4h EMA50), go long on break above daily R1.
In 4h downtrend (price below 4h EMA50), go short on break below daily S1.
Exit when price returns to daily Pivot or reverses trend.
Uses 4h/1d for signal direction, 1h only for entry timing.
Session filter (08-20 UTC) reduces noise trades.
Target: 15-37 trades/year (60-150 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
Volume filter ensures breakouts have participation, reducing false signals.
"""

name = "1h_4d_1d_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1h OHLCV
    close_1h = prices['close'].values
    high_1h = prices['high'].values
    low_1h = prices['low'].values
    volume_1h = prices['volume'].values
    
    # --- Daily Camarilla Levels (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R1, S1, Pivot (using previous day's OHLC)
    camarilla_R1 = np.full(len(high_1d), np.nan)
    camarilla_S1 = np.full(len(high_1d), np.nan)
    camarilla_P = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        # Camarilla formulas
        camarilla_P[i] = (ph + pl + pc) / 3.0
        camarilla_R1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    # --- 4h Trend Filter: EMA50 ---
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # --- Volume Confirmation: 1h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_1h).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close_1h[i] > ema50_4h_aligned[i]
        trend_down = close_1h[i] < ema50_4h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_1h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume
            if trend_up and vol_ok and close_1h[i] > camarilla_R1_aligned[i]:
                # Long: 4h uptrend + volume + break above daily R1
                signals[i] = 0.20
                position = 1
            elif trend_down and vol_ok and close_1h[i] < camarilla_S1_aligned[i]:
                # Short: 4h downtrend + volume + break below daily S1
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to daily Pivot
                if not trend_up or close_1h[i] < camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: trend turns up OR price returns to daily Pivot
                if not trend_down or close_1h[i] > camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals