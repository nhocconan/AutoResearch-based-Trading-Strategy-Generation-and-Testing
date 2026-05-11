#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_1dTrend_Volume
Hypothesis: Camarilla R1/S1 breakout with 1d EMA34 trend filter and volume confirmation.
- Long when: Close breaks above R1, 1d EMA34 up, volume > 20-period average
- Short when: Close breaks below S1, 1d EMA34 down, volume > 20-period average
- Exit when price returns to Camarilla pivot point (PP) or trend reverses
Camarilla levels provide precise intraday support/resistance. Trend filter ensures
we trade with higher timeframe momentum. Volume confirms participation.
Targets 20-35 trades/year (80-140 over 4 years) to minimize fee drag.
"""

name = "4h_1d_Camarilla_R1S1_Breakout_1dTrend_Volume"
timeframe = "4h"
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
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA34 ---
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- Camarilla Levels from 1d OHLC ---
    # Calculate once per 1d bar, then align to 4h
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_for_camarilla = df_1d['close'].values
    
    # Camarilla formula
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pp_1d = (high_1d + low_1d + close_1d_for_camarilla) / 3.0
    r1_1d = close_1d_for_camarilla + (high_1d - low_1d) * 1.1 / 12.0
    s1_1d = close_1d_for_camarilla - (high_1d - low_1d) * 1.1 / 12.0
    
    # Align to 4h
    pp_1d_aligned = align_htf_to_ltf(prices, df_1d, pp_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40  # for EMA34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(pp_1d_aligned[i]) or 
            np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_4h[i] > ema34_1d_aligned[i]
        trend_down = close_4h[i] < ema34_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1d trend with volume
            if close_4h[i] > r1_1d_aligned[i] and trend_up and vol_ok:
                # Long: break above R1 + 1d uptrend + volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < s1_1d_aligned[i] and trend_down and vol_ok:
                # Short: break below S1 + 1d downtrend + volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price returns to PP OR trend turns down
                if close_4h[i] <= pp_1d_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to PP OR trend turns up
                if close_4h[i] >= pp_1d_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals