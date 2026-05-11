#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Bounce_TrendFilter
Hypothesis: Use 1d Camarilla pivot levels (S1/S2 for long, R1/R2 for short) on 12h timeframe with 1d trend filter (EMA50) and volume confirmation.
- Long when: price crosses above S1, 1d EMA50 uptrend, volume > 20-period average
- Short when: price crosses below R1, 1d EMA50 downtrend, volume > 20-period average
- Exit when price crosses back through pivot level or trend reverses
Camarilla pivots identify key intraday support/resistance. Works in range-bound markets (bounce off S1/R1) and trends (break S2/R2 with trend).
Targets 12-30 trades/year (48-120 over 4 years) to minimize fee drag.
"""

name = "12h_1d_Camarilla_Pivot_Bounce_TrendFilter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Trend Filter: EMA50 ---
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # --- 1d Camarilla Pivot Levels (based on previous day) ---
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Camarilla levels: S1 = CP - (H-L)*1.1/6, S2 = CP - (H-L)*1.1/4
    # R1 = CP + (H-L)*1.1/6, R2 = CP + (H-L)*1.1/4
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    camarilla_p = typical_price.values
    camarilla_range = high_1d - low_1d
    
    s1 = camarilla_p - camarilla_range * 1.1 / 6
    s2 = camarilla_p - camarilla_range * 1.1 / 4
    r1 = camarilla_p + camarilla_range * 1.1 / 6
    r2 = camarilla_p + camarilla_range * 1.1 / 4
    
    # Align pivot levels to 12h timeframe (use previous day's close for calculation)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    
    # --- Volume Confirmation: 12h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30  # for EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1d trend
        trend_up = close_12h[i] > ema50_1d_aligned[i]
        trend_down = close_12h[i] < ema50_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_12h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries: price crosses S1 (long) or R1 (short) with trend and volume
            # Long: price crosses above S1, 1d uptrend, volume confirmation
            if (close_12h[i] > s1_aligned[i] and close_12h[i-1] <= s1_aligned[i-1] and 
                trend_up and vol_ok):
                signals[i] = 0.25
                position = 1
            # Short: price crosses below R1, 1d downtrend, volume confirmation
            elif (close_12h[i] < r1_aligned[i] and close_12h[i-1] >= r1_aligned[i-1] and 
                  trend_down and vol_ok):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below S1 OR trend turns down
                if (close_12h[i] < s1_aligned[i] and close_12h[i-1] >= s1_aligned[i-1]) or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above R1 OR trend turns up
                if (close_12h[i] > r1_aligned[i] and close_12h[i-1] <= r1_aligned[i-1]) or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals