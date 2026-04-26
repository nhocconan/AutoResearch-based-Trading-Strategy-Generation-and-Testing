#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: Camarilla R1/S1 breakout with 1-week trend filter and volume confirmation captures 
institutional moves on 12h timeframe. Go long when price breaks above R1 with 1w uptrend and 
volume spike. Go short when price breaks below S1 with 1w downtrend and volume spike. 
Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_ = prices['open'].values
    
    # Get weekly data for trend filter (using 1w as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Get daily data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Camarilla: R4 = close + 1.5*(high-low), R3 = close + 1.125*(high-low), 
    # R2 = close + 0.75*(high-low), R1 = close + 0.5*(high-low)
    # S1 = close - 0.5*(high-low), S2 = close - 0.75*(high-low), 
    # S3 = close - 1.125*(high-low), S4 = close - 1.5*(high-low)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values (shift by 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    # First day: use same day's values (no look-ahead)
    prev_high_1d[0] = high_1d[0]
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    # Calculate Camarilla R1 and S1
    camarilla_R1 = prev_close_1d + 0.5 * (prev_high_1d - prev_low_1d)
    camarilla_S1 = prev_close_1d - 0.5 * (prev_high_1d - prev_low_1d)
    
    # Align to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    
    # Volume confirmation: volume > 2.0x 24-period MA (24*12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 24 for volume MA)
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend and volume confirmation
            if close[i] > camarilla_R1_aligned[i] and uptrend_1w[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1w downtrend and volume confirmation
            elif close[i] < camarilla_S1_aligned[i] and downtrend_1w[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R1 OR 1w trend changes to downtrend
            if close[i] < camarilla_R1_aligned[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S1 OR 1w trend changes to uptrend
            if close[i] > camarilla_S1_aligned[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0