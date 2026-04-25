#!/usr/bin/env python3
"""
1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike
Hypothesis: On 1h timeframe, trade Camarilla R1/S1 breakouts in direction of 4h EMA34 trend with volume spike confirmation.
Uses discrete position sizing (0.20) to limit fee drag. Targets 15-30 trades/year.
Works in bull markets (breakouts with trend) and bear markets (fades from extremes with volume).
Session filter (08-20 UTC) reduces noise trades.
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
    
    # Get 4h data for EMA34 trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h EMA34 for trend filter
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Camarilla levels: R1/S1 from 4h
    camarilla_r1 = close_4h + (high_4h - low_4h) * 1.1 / 12
    camarilla_s1 = close_4h - (high_4h - low_4h) * 1.1 / 12
    
    # Align to 1h timeframe (completed 4h bar only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34 (34), volume MA (20), Camarilla (0)
    start_idx = max(34, 20, 0)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above R1 + 4h uptrend + volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema_34_4h_aligned[i]) and \
                         volume_spike[i]
            # Short: price closes below S1 + 4h downtrend + volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema_34_4h_aligned[i]) and \
                          volume_spike[i]
            
            if long_setup:
                signals[i] = 0.20
                position = 1
            elif short_setup:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit: price closes below S1 OR 4h trend turns down
            if (close[i] < camarilla_s1_aligned[i]) or \
               (close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit: price closes above R1 OR 4h trend turns up
            if (close[i] > camarilla_r1_aligned[i]) or \
               (close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1S1_Breakout_4hTrend_VolumeSpike"
timeframe = "1h"
leverage = 1.0