#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Camarilla pivot levels (R1, S1) from 1d timeframe act as support/resistance. 
Breakout above R1 with 1d uptrend and volume spike (>1.5x 20-period MA) = long.
Breakdown below S1 with 1d downtrend and volume spike = short.
Uses discrete position sizing (0.25) to minimize fee churn.
Designed to work in both bull and bear markets by following the 1d trend.
Target: 19-50 trades/year (75-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels (R1, S1) from previous 1d bar
    # Camarilla: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    rng = high_1d - low_1d
    camarilla_r1 = close_1d + 1.1 * rng / 12
    camarilla_s1 = close_1d - 1.1 * rng / 12
    
    # Align to 4h timeframe (use previous day's pivots)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # 1d EMA34 trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    uptrend_1d = close > ema_34_1d_aligned
    downtrend_1d = close < ema_34_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 34 for 1d EMA + 2 for pivots + 20 for volume MA)
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: break above R1 with 1d uptrend and volume spike
            if (close[i] > camarilla_r1_aligned[i] and 
                uptrend_1d[i] and volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with 1d downtrend and volume spike
            elif (close[i] < camarilla_s1_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price re-enters Camarilla H-L range (mean reversion) OR 1d trend changes
            camarilla_h = camarilla_r1_aligned[i] - 1.1 * (high_1d[i] - low_1d[i]) / 12  # Approximate HL midpoint
            camarilla_l = camarilla_s1_aligned[i] + 1.1 * (high_1d[i] - low_1d[i]) / 12
            if (close[i] < camarilla_r1_aligned[i] and close[i] > camarilla_s1_aligned[i]) or \
               (not uptrend_1d[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price re-enters Camarilla H-L range OR 1d trend changes
            camarilla_h = camarilla_r1_aligned[i] - 1.1 * (high_1d[i] - low_1d[i]) / 12
            camarilla_l = camarilla_s1_aligned[i] + 1.1 * (high_1d[i] - low_1d[i]) / 12
            if (close[i] < camarilla_r1_aligned[i] and close[i] > camarilla_s1_aligned[i]) or \
               (not downtrend_1d[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0