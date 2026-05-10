#!/usr/bin/env python3
# 12h_1w_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Trade 12h Camarilla R1/S1 breakouts aligned with 1w trend and 1d volume confirmation.
# In bull markets (1w trend up), buy breaks above R1 with volume confirmation.
# In bear markets (1w trend down), sell breaks below S1 with volume confirmation.
# Uses 1d volume spike (1.5x average) to confirm institutional participation.
# Target: 15-25 trades/year per symbol to avoid fee drag.

name = "12h_1w_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # 12h close series for calculations
    close_s = pd.Series(close)
    volume_s = pd.Series(volume)
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # Use 50-period EMA for 1w trend
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_uptrend = close_1w > ema50_1w
    weekly_downtrend = close_1w < ema50_1w
    
    # Align weekly trend to 12h
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    
    # 1d data for Camarilla levels and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Camarilla levels for R1 and S1
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d)
    r1_level = close_1d + (1.1 * camarilla_range / 12)
    s1_level = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 12h
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_level)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_level)
    
    # 1d volume average (20-period) for confirmation
    volume_1d_s = pd.Series(volume_1d)
    vol_ma_1d = volume_1d_s.rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        
        # Volume confirmation: current 12h volume > 1.5x 1d average volume
        vol_ratio = volume[i] / vol_ma_1d_aligned[i] if vol_ma_1d_aligned[i] > 0 else 0
        volume_confirm = vol_ratio > 1.5
        
        if position == 0:
            # Enter long: weekly uptrend + price breaks above R1 + volume confirmation
            if weekly_up and volume_confirm:
                if close[i] > r1_aligned[i]:
                    signals[i] = 0.25
                    position = 1
            # Enter short: weekly downtrend + price breaks below S1 + volume confirmation
            elif weekly_down and volume_confirm:
                if close[i] < s1_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit long: weekly trend changes or price breaks below S1 (reversal)
            if not weekly_up or close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: weekly trend changes or price breaks above R1 (reversal)
            if not weekly_down or close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals