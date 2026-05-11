#!/usr/bin/env python3
"""
4h_1d_1w_Camarilla_R1_S1_Breakout_1wTrend_Volume
Hypothesis: Camarilla R1/S1 breakouts on 4h with 1w trend filter and volume confirmation.
- Long when: price breaks above R1, 1w EMA34 uptrend, volume > 20-period average
- Short when: price breaks below S1, 1w EMA34 downtrend, volume > 20-period average
- Exit when price crosses H4/L4 (Camarilla midpoint) or trend reverses
Camarilla levels provide institutional support/resistance. Weekly trend filters ensure
we trade with the dominant momentum. Volume confirms institutional participation.
Targets 20-35 trades/year (80-140 over 4 years) to minimize fee drag.
Works in bull by riding uptrends from R1 breakouts, in bear by catching downtrends from S1 breaks.
"""

name = "4h_1d_1w_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels for given high, low, close"""
    range_val = high - low
    if range_val == 0:
        return close, close, close, close, close, close
    
    H4 = close + range_val * 1.1 / 2
    H3 = close + range_val * 1.1 / 4
    H2 = close + range_val * 1.1 / 6
    H1 = close + range_val * 1.1 / 12
    L1 = close - range_val * 1.1 / 12
    L2 = close - range_val * 1.1 / 6
    L3 = close - range_val * 1.1 / 4
    L4 = close - range_val * 1.1 / 2
    
    return H4, H3, H2, H1, L1, L2, L3, L4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 4h OHLCV
    close_4h = prices['close'].values
    high_4h = prices['high'].values
    low_4h = prices['low'].values
    volume_4h = prices['volume'].values
    
    # --- 1w Trend Filter: EMA34 ---
    close_1w = df_1w['close'].values
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # --- Camarilla Levels from 1d (previous day) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for today's Camarilla levels
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla for each day, then align to 4h
    H4_1d = np.full_like(prev_close, np.nan)
    H1_1d = np.full_like(prev_close, np.nan)
    L1_1d = np.full_like(prev_close, np.nan)
    L4_1d = np.full_like(prev_close, np.nan)
    
    for i in range(len(prev_close)):
        if not (np.isnan(prev_high[i]) or np.isnan(prev_low[i]) or np.isnan(prev_close[i])):
            H4, H1, L1, L4, _, _, _, _ = calculate_camarilla(prev_high[i], prev_low[i], prev_close[i])
            H4_1d[i] = H4
            H1_1d[i] = H1
            L1_1d[i] = L1
            L4_1d[i] = L4
    
    H4_1d_aligned = align_htf_to_ltf(prices, df_1d, H4_1d)
    H1_1d_aligned = align_htf_to_ltf(prices, df_1d, H1_1d)
    L1_1d_aligned = align_htf_to_ltf(prices, df_1d, L1_1d)
    L4_1d_aligned = align_htf_to_ltf(prices, df_1d, L4_1d)
    
    # --- Volume Confirmation: 4h volume > 20-period average ---
    vol_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period (need 1w EMA34 and Camarilla calculation)
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema34_1w_aligned[i]) or np.isnan(H4_1d_aligned[i]) or 
            np.isnan(L4_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine 1w trend
        trend_up = close_4h[i] > ema34_1w_aligned[i]
        trend_down = close_4h[i] < ema34_1w_aligned[i]
        
        # Volume confirmation
        vol_ok = volume_4h[i] > vol_ma_20[i]
        
        if position == 0:
            # Look for entries only in direction of 1w trend with volume
            if close_4h[i] > H1_1d_aligned[i] and trend_up and vol_ok:
                # Long: price above H1 (breakout), 1w uptrend, volume
                signals[i] = 0.25
                position = 1
            elif close_4h[i] < L1_1d_aligned[i] and trend_down and vol_ok:
                # Short: price below L1 (breakdown), 1w downtrend, volume
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit long: price crosses below H4 (midpoint) OR trend turns down
                if close_4h[i] < H4_1d_aligned[i] or not trend_up:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above L4 (midpoint) OR trend turns up
                if close_4h[i] > L4_1d_aligned[i] or not trend_down:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals