#!/usr/bin/env python3
"""
4h_12h_Camarilla_R1_S1_Breakout_12hTrend_Volume
Hypothesis: Camarilla R1/S1 breakout on 4h with 12h EMA trend filter and volume confirmation.
Works in bull by riding breakouts with trend, in bear by avoiding false breakouts via trend filter.
Volume ensures breakout has participation. Targets ~25-40 trades/year (100-160 over 4 years).
"""

name = "4h_12h_Camarilla_R1_S1_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 12h Trend Filter: EMA50 ---
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # --- Camarilla Levels from Previous Day ---
    # Need daily high, low, close
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1 and S1: H = (H-L)*1.1/12 + C
    camarilla_R1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_S1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 4h timeframe (1 day = 6 four-hour bars)
    camarilla_R1_4h = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_4h = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50  # for EMA50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(camarilla_R1_4h[i]) or 
            np.isnan(camarilla_S1_4h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 12h trend using close vs EMA
        trend_up = close_4h[i] > ema50_12h_aligned[i]
        trend_down = close_4h[i] < ema50_12h_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for breakouts in direction of 12h trend with volume
            if close_4h[i] > camarilla_R1_4h[i] and trend_up and vol_ok:
                # Long breakout above R1 with uptrend and volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < camarilla_S1_4h[i] and trend_down and vol_ok:
                # Short breakdown below S1 with downtrend and volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: reverse signal or trend change
            if position == 1:
                # Exit long: price breaks below S1 OR trend turns down
                if close_4h[i] < camarilla_S1_4h[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price breaks above R1 OR trend turns up
                if close_4h[i] > camarilla_R1_4h[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals