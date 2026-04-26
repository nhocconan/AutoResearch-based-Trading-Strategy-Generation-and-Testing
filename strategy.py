#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter
Hypothesis: Weekly Camarilla R1/S1 levels act as strong support/resistance. Breakout above R1 with 1w uptrend and volume spike = long. Breakdown below S1 with 1w downtrend and volume spike = short. Uses 1d timeframe for execution, 1w for trend filter and pivot calculation. Target: 7-25 trades/year per symbol (30-100 total over 4 years). Works in bull (breakouts continue) and bear (fades at pivots provide short opportunities).
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
    
    # Get weekly data for Camarilla pivots and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = close_1w + (high_1w - low_1w) * 1.1 / 12
    camarilla_s1 = close_1w - (high_1w - low_1w) * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Weekly EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    uptrend_1w = close > ema_50_1w_aligned
    downtrend_1w = close < ema_50_1w_aligned
    
    # Volume confirmation: volume > 2.0x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA + 50 for EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: close breaks above R1, with 1w uptrend and volume spike
            if close[i] > camarilla_r1_aligned[i] and uptrend_1w[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: close breaks below S1, with 1w downtrend and volume spike
            elif close[i] < camarilla_s1_aligned[i] and downtrend_1w[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close drops below S1 (mean reversion) OR 1w trend changes to downtrend
            if close[i] < camarilla_s1_aligned[i] or not uptrend_1w[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close rises above R1 (mean reversion) OR 1w trend changes to uptrend
            if close[i] > camarilla_r1_aligned[i] or not downtrend_1w[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeFilter"
timeframe = "1d"
leverage = 1.0