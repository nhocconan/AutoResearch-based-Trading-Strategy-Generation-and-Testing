#!/usr/bin/env python3
# 12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
# Hypothesis: Trade Camarilla R1/S1 breakouts with weekly trend filter and volume confirmation.
# Long when price breaks above R1 with weekly uptrend and volume > 1.5x average.
# Short when price breaks below S1 with weekly downtrend and volume > 1.5x average.
# Uses weekly EMA50 for trend filter to capture major trends and avoid counter-trend trades.
# Target: 12-37 trades/year per symbol.

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Camarilla levels for 12h using previous bar's OHLC
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    hl_range = high - low
    r1 = close + 1.1 * hl_range / 12
    s1 = close - 1.1 * hl_range / 12
    
    # Shift to get previous bar's levels (no look-ahead)
    r1_prev = np.roll(r1, 1)
    s1_prev = np.roll(s1, 1)
    r1_prev[0] = np.nan
    s1_prev[0] = np.nan
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # Volume average (24-period ≈ 12 days)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_prev[i]) or np.isnan(s1_prev[i]) or np.isnan(vol_ma[i]) or
            np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above R1 + volume confirmation
            if weekly_up and volume_confirm:
                if close[i] > r1_prev[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + price breaks below S1 + volume confirmation
            elif weekly_down and volume_confirm:
                if close[i] < s1_prev[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit conditions: weekly trend reverses or price moves below R1
            if not weekly_up or close[i] < r1_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit conditions: weekly trend reverses or price moves above S1
            if not weekly_down or close[i] > s1_prev[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals