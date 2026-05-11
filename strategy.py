#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_R1_S1_Breakout_TrendVolume
Hypothesis: Use daily (1d) and weekly (1w) context to filter breakouts at daily Camarilla R1/S1 levels.
- In 1w uptrend (price above 1w EMA50), only take long breakouts above daily R1.
- In 1w downtrend (price below 1w EMA50), only take short breakouts below daily S1.
- Requires volume confirmation (4h volume > 20-period average).
- Exit when price returns to daily Pivot or weekly trend reverses.
Targets 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
"""

name = "4h_1d_1w_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Get daily and weekly data for context
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 2 or len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- Daily Camarilla Levels (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_R1 = np.full(len(high_1d), np.nan)
    camarilla_S1 = np.full(len(high_1d), np.nan)
    camarilla_P = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        camarilla_P[i] = (ph + pl + pc) / 3.0
        camarilla_R1[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align Daily Camarilla levels to 4h
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    # --- Weekly Trend Filter: EMA50 ---
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend
        weekly_uptrend = close_4h[i] > ema50_1w_aligned[i]
        weekly_downtrend = close_4h[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of weekly trend with volume
            if weekly_uptrend and vol_ok and close_4h[i] > camarilla_R1_aligned[i]:
                # Long: weekly uptrend + volume + break above daily R1
                signals[i] = 0.25
                position = 1
            elif weekly_downtrend and vol_ok and close_4h[i] < camarilla_S1_aligned[i]:
                # Short: weekly downtrend + volume + break below daily S1
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: weekly trend turns down OR price returns to daily Pivot
                if not weekly_uptrend or close_4h[i] < camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: weekly trend turns up OR price returns to daily Pivot
                if not weekly_downtrend or close_4h[i] > camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals