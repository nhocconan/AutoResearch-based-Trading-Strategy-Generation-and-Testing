#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1
Hypothesis: Camarilla R1/S1 breakouts on 1h with 4h EMA34 trend filter and volume spike confirmation.
Uses 4h for signal direction (trend) and 1h for precise entry timing. Volume spike (>2.0x 20-bar avg)
confirms breakout momentum. Discrete sizing (0.20) limits trades to ~20-40/year to avoid fee drag.
Designed to work in both bull and bear markets by following 4h trend while using 1h for timing.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for HTF trend filter and Camarilla levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate EMA34 on 4h close for trend filter
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # Calculate Camarilla levels from previous 4h bar (HLC of prior bar)
    camarilla_r1 = close_4h + 1.1 * (high_4h - low_4h) / 2
    camarilla_s1 = close_4h - 1.1 * (high_4h - low_4h) / 2
    
    # Align Camarilla levels to 1h timeframe (use previous bar's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s1)
    
    # Calculate 20-bar average volume for confirmation on 1h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for EMA34, volume MA20
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_4h_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_ma20[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        if position == 0:
            # Volume confirmation: current volume > 2.0x 20-bar average
            volume_confirm = volume[i] > 2.0 * vol_ma20[i]
            
            # Long: price breaks above Camarilla R1 in uptrend with volume spike
            # Short: price breaks below Camarilla S1 in downtrend with volume spike
            long_signal = (close[i] > camarilla_r1_aligned[i]) and (close[i] > ema34_4h_aligned[i]) and volume_confirm
            short_signal = (close[i] < camarilla_s1_aligned[i]) and (close[i] < ema34_4h_aligned[i]) and volume_confirm
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.20
            # Exit when price moves back below 4h EMA34 (trend reversal)
            exit_signal = close[i] < ema34_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.20
            # Exit when price moves back above 4h EMA34 (trend reversal)
            exit_signal = close[i] > ema34_4h_aligned[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_VolumeSpike_v1"
timeframe = "1h"
leverage = 1.0