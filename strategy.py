#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Camarilla R1/S1 levels act as strong intraday support/resistance. 
Breakout above R1 with volume spike and weekly uptrend = long. Breakdown below S1 with volume spike and weekly downtrend = short.
Uses weekly EMA34 trend filter to work in both bull/bear markets (only long in weekly uptrend, short in downtrend).
Volume spike filter ensures conviction. Target: 15-30 trades/year per symbol to minimize fee drag.
Timeframe: 1d, HTF: 1w
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla levels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Camarilla levels from weekly OHLC (use previous completed weekly bar)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w_shift = df_1w['close'].shift(1).values  # Previous bar close
    
    # Calculate Camarilla levels for each weekly bar
    rng = high_1w - low_1w
    camarilla_r1 = close_1w_shift + 1.1 * rng / 2  # R1 = Close + 1.1*(High-Low)/2
    camarilla_s1 = close_1w_shift - 1.1 * rng / 2  # S1 = Close - 1.1*(High-Low)/2
    
    # Align Camarilla levels to daily timeframe (with proper delay for completed bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1)
    
    # Volume spike detector (20-bar volume MA on daily)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from weekly EMA34
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume spike and weekly uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike and weekly downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close below R1 (failed breakout) OR weekly trend change to downtrend
            if close[i] < camarilla_r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close above S1 (failed breakdown) OR weekly trend change to uptrend
            if close[i] > camarilla_s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0