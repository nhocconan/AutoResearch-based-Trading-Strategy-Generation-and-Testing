#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: 12h timeframe strategy using Camarilla R1/S1 breakouts with 1-week EMA34 trend filter and volume confirmation (>1.5x 30-period MA).
Long when price breaks above R1 with 1w uptrend and volume spike.
Short when price breaks below S1 with 1w downtrend and volume filter.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to capture medium-term swings in both bull and bear markets by following the 1-week trend.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivot calculation (based on previous 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get 1w data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # 1w EMA34 trend filter (needs extra delay as EMA confirms after bar close)
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w, additional_delay_bars=1)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use previous day's OHLC (avoid look-ahead)
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan  # First value has no previous day
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    camarilla_range = (high_1d_prev - low_1d_prev) * 1.1 / 12
    r1 = close_1d_prev + camarilla_range
    s1 = close_1d_prev - camarilla_range
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5x 30-period MA
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1w EMA + 1 for 1d shift + 30 for volume MA)
    start_idx = 65
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 1w uptrend and volume spike
            if close[i] > r1_aligned[i] and uptrend_1w[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with 1w downtrend and volume spike
            elif close[i] < s1_aligned[i] and downtrend_1w[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S1 (mean reversion) OR 1w trend changes to downtrend
            if close[i] < s1_aligned[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R1 (mean reversion) OR 1w trend changes to uptrend
            if close[i] > r1_aligned[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "12h"
leverage = 1.0