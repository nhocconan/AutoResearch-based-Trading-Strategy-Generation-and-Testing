# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
12h_1d_Camarilla_R1_S1_Breakout_TrendVolume_v2
Hypothesis: Use daily Camarilla R1/S1 levels as breakout triggers with 12h trend filter (12h EMA50) and volume confirmation.
Reduce trade frequency by adding a minimum holding period (minimum 10 bars) and increasing volume threshold.
In 12h uptrend (price above 12h EMA50), go long on break above daily R1.
In 12h downtrend (price below 12h EMA50), go short on break below daily S1.
Exit when price returns to daily Pivot or reverses trend.
Target: 15-25 trades/year (60-100 over 4 years) to minimize fee drag.
Works in bull by buying breakouts in uptrend; works in bear by selling breakdowns in downtrend.
Volume filter ensures breakouts have participation, reducing false signals.
"""

name = "12h_1d_Camarilla_R1_S1_Breakout_TrendVolume_v2"
timeframe = "12h"
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
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
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
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_P_aligned = align_htf_to_ltf(prices, df_1d, camarilla_P)
    
    # --- 12h Trend Filter: EMA50 ---
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # --- Volume Confirmation: 12h volume > 30-period average (stricter) ---
    vol_ma_30 = pd.Series(volume_12h).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track bars in position for minimum hold
    
    # Start after warmup period
    start_idx = 50  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(camarilla_R1_aligned[i]) or np.isnan(camarilla_S1_aligned[i]) or 
            np.isnan(camarilla_P_aligned[i]) or np.isnan(ema50_12h[i]) or 
            np.isnan(vol_ma_30[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine 12h trend
        trend_up = close_12h[i] > ema50_12h[i]
        trend_down = close_12h[i] < ema50_12h[i]
        
        # Volume confirmation (stricter threshold)
        vol_ok = volume_12h[i] > vol_ma_30[i]
        
        # Increment bars in position
        if position != 0:
            bars_since_entry += 1
        
        if position == 0:
            # Look for entries only in direction of 12h trend with volume
            # Require minimum 10 bars since last exit (prevents whipsaw)
            if bars_since_entry >= 10 or bars_since_entry == 0:
                if trend_up and vol_ok and close_12h[i] > camarilla_R1_aligned[i]:
                    # Long: 12h uptrend + volume + break above daily R1
                    signals[i] = 0.25
                    position = 1
                    bars_since_entry = 0
                elif trend_down and vol_ok and close_12h[i] < camarilla_S1_aligned[i]:
                    # Short: 12h downtrend + volume + break below daily S1
                    signals[i] = -0.25
                    position = -1
                    bars_since_entry = 0
        else:
            # Exit conditions
            if position == 1:
                # Exit long: trend turns down OR price returns to daily Pivot
                if not trend_up or close_12h[i] < camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: trend turns up OR price returns to daily Pivot
                if not trend_down or close_12h[i] > camarilla_P_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals