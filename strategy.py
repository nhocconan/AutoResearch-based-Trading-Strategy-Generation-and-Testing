#!/usr/bin/env python3
"""
12h_1D_Camarilla_R1S1_Breakout_1D_EMA34_Trend
Hypothesis: Daily Camarilla R1/S1 levels act as key support/resistance. A breakout above R1 or below S1 with volume confirmation and aligned with daily EMA34 trend signals momentum continuation. This combines strong horizontal levels with trend filtering to work in both bull and bear markets. Targets 50-150 total trades over 4 years on 12h timeframe.
"""

name = "12h_1D_Camarilla_R1S1_Breakout_1D_EMA34_Trend"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and EMA34 trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d Daily OHLC for Camarilla calculation ---
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_ = prev_high - prev_low
    camarilla_pivot = (prev_high + prev_low + prev_close) / 3.0
    r1 = camarilla_pivot + (range_ * 1.1 / 12)
    s1 = camarilla_pivot - (range_ * 1.1 / 12)
    r2 = camarilla_pivot + (range_ * 1.1 / 6)
    s2 = camarilla_pivot - (range_ * 1.1 / 6)
    
    # Align daily Camarilla levels to 12h
    r1_12h = align_htf_to_ltf(prices, df_1d, r1)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1)
    r2_12h = align_htf_to_ltf(prices, df_1d, r2)
    s2_12h = align_htf_to_ltf(prices, df_1d, s2)
    
    # --- 1d EMA34 for trend filter ---
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # --- 12h Volume confirmation ---
    vol_avg_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup
    start_idx = 40  # for EMA34 and volume average
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or 
            np.isnan(ema34_12h[i]) or np.isnan(vol_avg_12h[i])):
            if position != 0:
                # Simple stop: exit if price crosses Camarilla S2/R2
                if position == 1 and close_12h[i] <= s2_12h[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= r2_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Volume confirmation: current volume > 1.5x 12h average
        vol_confirm = volume_12h[i] > 1.5 * vol_avg_12h[i]
        
        if position == 0:
            # Look for breakout entries
            if vol_confirm:
                # Long breakout above R1 with EMA34 uptrend
                if close_12h[i] > r1_12h[i] and close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close_12h[i]
                # Short breakdown below S1 with EMA34 downtrend
                elif close_12h[i] < s1_12h[i] and close_12h[i] < ema34_12h[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close_12h[i]
        else:
            # Manage existing position
            if position == 1:
                # Long: exit if price breaks below S1 or reverses below EMA34
                if close_12h[i] < s1_12h[i] or close_12h[i] < ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short: exit if price breaks above R1 or reverses above EMA34
                if close_12h[i] > r1_12h[i] or close_12h[i] > ema34_12h[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals