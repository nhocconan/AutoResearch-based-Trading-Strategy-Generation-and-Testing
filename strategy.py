#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike
Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter (price > 4h EMA34) and volume confirmation (>1.5x 24-bar avg).
Enters long when price breaks above R1 in 4h uptrend, short when breaks below S1 in 4h downtrend.
Uses discrete sizing (0.20) to limit fee churn. Target: 60-150 total trades over 4 years.
Designed for 1h timeframe with session filter (08-20 UTC) to reduce noise.
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
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # 4h data for HTF trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Previous day's high, low, close for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_range = high_1d - low_1d
    r1 = close_1d + (1.1 * camarilla_range / 12)
    s1 = close_1d - (1.1 * camarilla_range / 12)
    
    # Align Camarilla levels to 1h (previous day's levels available at 00:00 UTC)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume spike: current volume > 1.5x 24-period average
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need 24-period volume MA and Camarilla data
    start_idx = 24
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        
        if position == 0:
            # Long: price breaks above R1 in 4h uptrend with volume confirmation
            bullish_breakout = (curr_close > r1_aligned[i]) and \
                              (close[i] > ema_34_4h_aligned[i]) and \
                              volume_spike[i]
            # Short: price breaks below S1 in 4h downtrend with volume confirmation
            bearish_breakout = (curr_close < s1_aligned[i]) and \
                              (close[i] < ema_34_4h_aligned[i]) and \
                              volume_spike[i]
            
            if bullish_breakout:
                signals[i] = 0.20
                position = 1
            elif bearish_breakout:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price breaks below S1 OR trend turns down
            if (curr_close < s1_aligned[i]) or \
               (close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price breaks above R1 OR trend turns up
            if (curr_close > r1_aligned[i]) or \
               (close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0