#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeFilter
Hypothesis: Weekly EMA34 trend + daily Camarilla R1/S1 breakouts with volume confirmation on 12h timeframe.
Long: Price breaks above R1 with weekly uptrend and volume spike. Short: Price breaks below S1 with weekly downtrend and volume spike.
Exit: Mean reversion to opposite Camarilla level or trend change. Designed for fewer trades (target 12-37/year) to minimize fee drag on 12h chart.
Works in bull (trend continuation) and bear (mean reversion at extreme pivots during range-bound periods).
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get weekly data for EMA34 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    uptrend_1w = close > ema_34_1w_aligned
    downtrend_1w = close < ema_34_1w_aligned
    
    # Volume confirmation: volume > 2.0x 30-period MA (longer MA for 12h to reduce noise)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 30 for volume MA + 34 for weekly EMA)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
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
            # Long: close breaks above R1, with weekly uptrend and volume spike
            if close[i] > camarilla_r1_aligned[i] and uptrend_1w[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1, with weekly downtrend and volume spike
            elif close[i] < camarilla_s1_aligned[i] and downtrend_1w[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below S1 (mean reversion) OR weekly trend changes to downtrend
            if close[i] < camarilla_s1_aligned[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above R1 (mean reversion) OR weekly trend changes to uptrend
            if close[i] > camarilla_r1_aligned[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wEMA34_Trend_VolumeFilter"
timeframe = "12h"
leverage = 1.0