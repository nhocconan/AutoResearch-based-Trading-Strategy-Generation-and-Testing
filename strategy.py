#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1, S1) from 1d data act as intraday support/resistance on 12h timeframe.
Breakout above R1 with 1w uptrend and volume spike (>2x 24-period MA) signals long.
Breakdown below S1 with 1w downtrend and volume spike signals short.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the 1w trend, which adapts to regime changes.
Target: 12-37 trades/year (50-150 total over 4 years).
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
    
    # Get 1d data for Camarilla pivot calculation (based on previous day)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d[0] = np.nan  # First value has no previous day
    prev_high_1d[0] = np.nan
    prev_low_1d[0] = np.nan
    
    # Calculate Camarilla R1 and S1 for previous day
    camarilla_r1_1d = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 12
    camarilla_s1_1d = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 12
    
    # Align 1d Camarilla levels to 12h timeframe (wait for 1d bar to close)
    camarilla_r1_12h = align_htf_to_ltf(prices, df_1d, camarilla_r1_1d)
    camarilla_s1_12h = align_htf_to_ltf(prices, df_1d, camarilla_s1_1d)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 1w EMA34 trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Volume confirmation: volume > 2.0x 24-period MA (2*12h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA + 1 for 1d shift + 24 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_12h[i]) or np.isnan(camarilla_s1_12h[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: break above R1 with 1w uptrend and volume spike
            if (close[i] > camarilla_r1_12h[i] and 
                uptrend_1w[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1w downtrend and volume spike
            elif (close[i] < camarilla_s1_12h[i] and 
                  downtrend_1w[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price returns below R1 OR 1w trend changes to downtrend
            if (close[i] < camarilla_r1_12h[i] or not uptrend_1w[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price returns above S1 OR 1w trend changes to uptrend
            if (close[i] > camarilla_s1_12h[i] or not downtrend_1w[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0