#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Trade 12h Camarilla R1/S1 breakouts in direction of 1d trend (EMA34) with volume confirmation.
Camarilla levels from 1d provide robust support/resistance. 1d EMA34 trend filter ensures alignment with higher timeframe momentum.
Volume spike confirms breakout validity. Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
Works in bull markets (breakouts with trend) and bear markets (breakouts with trend, short bias).
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
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    typical_price_1d = (df_1d['high'].values + df_1d['low'].values + df_1d['close'].values) / 3
    range_1d = df_1d['high'].values - df_1d['low'].values
    
    # Camarilla R1 and S1 levels (main breakout levels)
    camarilla_r1 = typical_price_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = typical_price_1d - (range_1d * 1.1 / 12)
    
    # Align Camarilla levels to 12h timeframe (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for 1d EMA34 (34) and ensure Camarilla data is ready
    start_idx = max(34, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 2.0 * average of last 10 periods (12h timeframe)
        if i >= 10:
            vol_avg = np.mean(volume[i-10:i])
            volume_spike = volume[i] > 2.0 * vol_avg
        else:
            volume_spike = False
        
        if position == 0:
            # Long: price breaks above R1 AND 1d trend bullish (close > EMA34) AND volume spike
            long_setup = (close[i] > camarilla_r1_aligned[i]) and \
                         (close[i] > ema_34_1d_aligned[i]) and \
                         volume_spike
            # Short: price breaks below S1 AND 1d trend bearish (close < EMA34) AND volume spike
            short_setup = (close[i] < camarilla_s1_aligned[i]) and \
                          (close[i] < ema_34_1d_aligned[i]) and \
                          volume_spike
            
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
            # Exit: price re-enters between S1 and R1 OR 1d trend turns bearish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] < ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price re-enters between S1 and R1 OR 1d trend turns bullish
            if (camarilla_s1_aligned[i] < close[i] < camarilla_r1_aligned[i]) or \
               (close[i] > ema_34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0