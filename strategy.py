#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike
Hypothesis: 4-hour Camarilla R1/S1 levels act as strong intraday support/resistance.
Breakout above R1 with volume spike and 12-hour uptrend = long. Breakdown below S1 with volume spike and 12-hour downtrend = short.
Uses 12h EMA34 trend filter to work in both bull/bear markets (only long in 12h uptrend, short in downtrend).
Volume spike filter ensures conviction. Target: 20-50 trades/year per symbol to minimize fee drag.
Timeframe: 4h, HTF: 12h
"""

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
    
    # Get 12h data for Camarilla levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from 12h OHLC (use previous completed 12h bar)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_shift = df_12h['close'].shift(1).values  # Previous bar close
    
    # Calculate Camarilla levels for each 12h bar
    rng = high_12h - low_12h
    camarilla_r1 = close_12h_shift + 1.1 * rng / 2  # R1 = Close + 1.1*(High-Low)/2
    camarilla_s1 = close_12h_shift - 1.1 * rng / 2  # S1 = Close - 1.1*(High-Low)/2
    
    # Align Camarilla levels to 4h timeframe (with proper delay for completed bar)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s1)
    
    # Volume spike detector (20-bar volume MA on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter from 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long: Break above R1 with volume spike and 12h uptrend
            if close[i] > camarilla_r1_aligned[i] and volume_spike[i] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Break below S1 with volume spike and 12h downtrend
            elif close[i] < camarilla_s1_aligned[i] and volume_spike[i] and downtrend:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Close below R1 (failed breakout) OR 12h trend change to downtrend
            if close[i] < camarilla_r1_aligned[i] or not uptrend:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Close above S1 (failed breakdown) OR 12h trend change to uptrend
            if close[i] > camarilla_s1_aligned[i] or not downtrend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0