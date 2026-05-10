#!/usr/bin/env python3
# 4H_Camarilla_R1_S1_Breakout_1dTrend_Volume_Confluence
# Hypothesis: Trade Camarilla pivot breakouts (R1/S1) with 1d trend filter and volume confirmation.
# Long when price breaks above R1 in uptrend (1d EMA50) with volume > 1.5x average.
# Short when price breaks below S1 in downtrend (1d EMA50) with volume > 1.5x average.
# Uses 4h Camarilla levels derived from previous 1d OHLC. Only trades in direction of higher timeframe trend.
# Designed to work in both bull and bear markets by following the 1d trend and using volume to confirm institutional participation.
# Target: 20-40 trades/year per symbol.

name = "4H_Camarilla_R1_S1_Breakout_1dTrend_Volume_Confluence"
timeframe = "4h"
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
    
    # Volume average (20-period)
    volume_s = pd.Series(volume)
    vol_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Daily data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # where C = (H+L+C)/3 (typical price)
    # But standard Camarilla uses: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    # We'll use the standard formula
    h_1d = df_1d['high'].values
    l_1d = df_1d['low'].values
    c_1d = df_1d['close'].values
    
    # Typical price for Camarilla reference (some use close, some use typical)
    # Using close as pivot point for simplicity
    pivot_1d = c_1d
    range_1d = h_1d - l_1d
    
    r1_1d = pivot_1d + range_1d * 1.1 / 12
    s1_1d = pivot_1d - range_1d * 1.1 / 12
    
    # Align Camarilla levels to 4h (these levels are valid for the entire day)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Daily trend filter: EMA50
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_uptrend = close_1d > ema50_1d
    daily_downtrend = close_1d < ema50_1d
    
    # Align daily trend to 4h
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data for EMA50
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(daily_uptrend_aligned[i]) or np.isnan(daily_downtrend_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        daily_up = daily_uptrend_aligned[i] > 0.5
        daily_down = daily_downtrend_aligned[i] > 0.5
        
        if position == 0:
            # Enter long: daily uptrend + price breaks above R1 + volume confirmation
            if daily_up and volume_confirm:
                if close[i] > r1_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: daily downtrend + price breaks below S1 + volume confirmation
            elif daily_down and volume_confirm:
                if close[i] < s1_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: trend reverses or price breaks below S1 (contrarian signal)
            if not daily_up or close[i] < s1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend reverses or price breaks above R1 (contrarian signal)
            if not daily_down or close[i] > r1_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals