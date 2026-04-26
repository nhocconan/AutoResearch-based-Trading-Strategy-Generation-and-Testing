#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend
Hypothesis: Camarilla pivot R1/S1 breakout on 4h with volume confirmation (>1.5x 20-period MA) and 1d EMA34 trend filter. 
Long when price breaks above R1 with volume spike and 1d uptrend. Short when price breaks below S1 with volume spike and 1d downtrend. 
Uses discrete position sizing (0.25) to minimize fee churn. 
Designed to capture intraday momentum within the higher timeframe trend, working in both bull and bear markets by following the 1d trend.
Target: 19-50 trades/year (75-200 total over 4 years).
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
    
    # Get 1d data for EMA34 trend filter and Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 1d EMA34 trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Previous day's OHLC for Camarilla pivot levels (using aligned 1d data)
    prev_close_1d = df_1d['close'].shift(1).values
    prev_high_1d = df_1d['high'].shift(1).values
    prev_low_1d = df_1d['low'].shift(1).values
    prev_open_1d = df_1d['open'].shift(1).values
    
    # Camarilla pivot levels calculation
    # R1 = Close + (High - Low) * 1.1/12
    # S1 = Close - (High - Low) * 1.1/12
    camarilla_range = (prev_high_1d - prev_low_1d) * 1.1 / 12
    r1 = prev_close_1d + camarilla_range
    s1 = prev_close_1d - camarilla_range
    
    # Align Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA + 20 for volume MA + 1 for shift)
    start_idx = 55
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(r1_aligned[i]) or 
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
            # Long: price breaks above R1 with volume spike and 1d uptrend
            if close[i] > r1_aligned[i] and volume_spike[i] and uptrend_1d[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and 1d downtrend
            elif close[i] < s1_aligned[i] and volume_spike[i] and downtrend_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price breaks below S1 OR 1d trend changes to downtrend
            if close[i] < s1_aligned[i] or not uptrend_1d[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price breaks above R1 OR 1d trend changes to uptrend
            if close[i] > r1_aligned[i] or not downtrend_1d[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0