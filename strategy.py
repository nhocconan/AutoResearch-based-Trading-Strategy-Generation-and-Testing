#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_R1_S1_Breakout_TrendVolume
Hypothesis: Use daily and weekly Camarilla R1/S1 levels as breakout triggers with 4h trend filter (4h EMA50) and volume confirmation.
In 4h uptrend (price above 4h EMA50), go long on break above daily or weekly R1.
In 4h downtrend (price below 4h EMA50), go short on break below daily or weekly S1.
Exit when price returns to daily or weekly Pivot or reverses trend.
Target: 20-40 trades/year (80-160 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
Volume filter ensures breakouts have participation, reducing false signals.
"""

name = "4h_1d_1w_Camarilla_R1_S1_Breakout_TrendVolume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get daily and weekly data for Camarilla levels
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
    
    camarilla_R1_1d = np.full(len(high_1d), np.nan)
    camarilla_S1_1d = np.full(len(high_1d), np.nan)
    camarilla_P_1d = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        # Camarilla formulas
        camarilla_P_1d[i] = (ph + pl + pc) / 3.0
        camarilla_R1_1d[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1_1d[i] = pc - (ph - pl) * 1.1 / 12
    
    # --- Weekly Camarilla Levels (from previous week) ---
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    camarilla_R1_1w = np.full(len(high_1w), np.nan)
    camarilla_S1_1w = np.full(len(high_1w), np.nan)
    camarilla_P_1w = np.full(len(high_1w), np.nan)
    
    for i in range(1, len(high_1w)):
        # Previous week's OHLC
        ph = high_1w[i-1]
        pl = low_1w[i-1]
        pc = close_1w[i-1]
        # Camarilla formulas
        camarilla_P_1w[i] = (ph + pl + pc) / 3.0
        camarilla_R1_1w[i] = pc + (ph - pl) * 1.1 / 12
        camarilla_S1_1w[i] = pc - (ph - pl) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_R1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1_1d)
    camarilla_S1_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1_1d)
    camarilla_P_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P_1d)
    
    camarilla_R1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_R1_1w)
    camarilla_S1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_S1_1w)
    camarilla_P_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_P_1w)
    
    # --- 4h Trend Filter: EMA50 ---
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_1d_aligned[i]) or np.isnan(camarilla_S1_1d_aligned[i]) or 
            np.isnan(camarilla_P_1d_aligned[i]) or np.isnan(camarilla_R1_1w_aligned[i]) or 
            np.isnan(camarilla_S1_1w_aligned[i]) or np.isnan(camarilla_P_1w_aligned[i]) or 
            np.isnan(ema50_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 4h trend
        trend_up = close_4h[i] > ema50_4h[i]
        trend_down = close_4h[i] < ema50_4h[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 4h trend with volume
            if trend_up and vol_ok and (close_4h[i] > camarilla_R1_1d_aligned[i] or close_4h[i] > camarilla_R1_1w_aligned[i]):
                # Long: 4h uptrend + volume + break above daily or weekly R1
                signals[i] = 0.25
                position = 1
            elif trend_down and vol_ok and (close_4h[i] < camarilla_S1_1d_aligned[i] or close_4h[i] < camarilla_S1_1w_aligned[i]):
                # Short: 4h downtrend + volume + break below daily or weekly S1
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to daily or weekly Pivot
                if not trend_up or close_4h[i] < camarilla_P_1d_aligned[i] or close_4h[i] < camarilla_P_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price returns to daily or weekly Pivot
                if not trend_down or close_4h[i] > camarilla_P_1d_aligned[i] or close_4h[i] > camarilla_P_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals