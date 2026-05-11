#!/usr/bin/env python3
"""
6h_1d_Camarilla_R3_S3_Breakout_TrendVolume
Hypothesis: Use daily Camarilla R3/S3 levels as breakout triggers with 6h trend filter (6h EMA50) and volume confirmation.
In 6h uptrend (price above 6h EMA50), go long on break above daily R3.
In 6h downtrend (price below 6h EMA50), go short on break below daily S3.
Exit when price returns to daily Pivot or reverses trend.
Targets 12-30 trades/year (48-120 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
Volume filter ensures breakouts have participation, reducing false signals.
"""

name = "6h_1d_Camarilla_R3_S3_Breakout_TrendVolume"
timeframe = "6h"
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
    
    # 6h OHLCV
    close_6h = prices['close'].values
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    volume_6h = prices['volume'].values
    
    # --- Daily Camarilla Levels (from previous day) ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels: R3, S3, Pivot (using previous day's OHLC)
    camarilla_R3 = np.full(len(high_1d), np.nan)
    camarilla_S3 = np.full(len(high_1d), np.nan)
    camarilla_P = np.full(len(high_1d), np.nan)
    
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        # Camarilla formulas
        camarilla_P[i] = (ph + pl + pc) / 3.0
        camarilla_R3[i] = pc + (ph - pl) * 1.1 * 3 / 4
        camarilla_S3[i] = pc - (ph - pl) * 1.1 * 3 / 4
    
    # Align Camarilla levels to 6h timeframe
    camarilla_R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    camarilla_S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    # --- 6h Trend Filter: EMA50 ---
    ema50_6h = pd.Series(close_6h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume Confirmation: 6h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R3_aligned[i]) or np.isnan(camarilla_S3_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(ema50_6h[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 6h trend
        trend_up = close_6h[i] > ema50_6h[i]
        trend_down = close_6h[i] < ema50_6h[i]
        
        # Volume confirmation
        vol_ok = volume_6h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 6h trend with volume
            if trend_up and vol_ok and close_6h[i] > camarilla_R3_aligned[i]:
                # Long: 6h uptrend + volume + break above daily R3
                signals[i] = 0.25
                position = 1
            elif trend_down and vol_ok and close_6h[i] < camarilla_S3_aligned[i]:
                # Short: 6h downtrend + volume + break below daily S3
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to daily Pivot
                if not trend_up or close_6h[i] < camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price returns to daily Pivot
                if not trend_down or close_6h[i] > camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals