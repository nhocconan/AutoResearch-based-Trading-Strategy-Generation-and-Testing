#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm
Hypothesis: 12h Camarilla R1/S1 breakout with 1-day EMA34 trend filter and volume confirmation.
Long when price breaks above R1 with 1d uptrend and volume spike.
Short when price breaks below S1 with 1d downtrend and volume spike.
Camarilla levels provide intraday support/resistance based on prior day's range.
Volume confirms institutional participation. Works in bull/bear regimes via 1d trend filter.
Target: 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for HTF trend filter and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 1-day EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # Camarilla R1 = C + ((H-L) * 1.1/12)
    # Camarilla S1 = C - ((H-L) * 1.1/12)
    h1 = high_1d[:-1]  # previous day's high
    l1 = low_1d[:-1]   # previous day's low
    c1 = close_1d[:-1] # previous day's close
    
    camarilla_r1 = c1 + ((h1 - l1) * 1.1 / 12)
    camarilla_s1 = c1 - ((h1 - l1) * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (previous day's levels available at 12h open)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34(1d) and volume MA
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > R1 + 1d uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema_34_1d_aligned[i]) and volume_spike[i]
            # Short: price < S1 + 1d downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema_34_1d_aligned[i]) and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price < R1 OR 1d trend turns down
            if (close[i] < camarilla_r1_aligned[i]) or (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price > S1 OR 1d trend turns up
            if (close[i] > camarilla_s1_aligned[i]) or (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0