#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v5
Hypothesis: On 12h timeframe, trade Camarilla R1/S1 breakouts from prior 12h bar with 1d EMA34 trend filter and volume spike confirmation. Target 12-37 trades/year by requiring confluence of HTF trend alignment and volume expansion. Designed to work in both bull and bear markets via trend filter - in bull markets take longs above EMA34, in bear markets take shorts below EMA34. Volume spike confirms institutional participation reducing false breakouts.
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
    
    # Get 1d data for HTF trend (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Get 12h data for Camarilla levels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous 12h bar (HLC of prior 12h)
    cam_high = pd.Series(df_12h['high'].values).shift(1).values
    cam_low = pd.Series(df_12h['low'].values).shift(1).values
    cam_close = pd.Series(df_12h['close'].values).shift(1).values
    
    # Camarilla R1, S1 levels (core breakout levels)
    R1 = cam_close + (cam_high - cam_low) * 1.1 / 12
    S1 = cam_close - (cam_high - cam_low) * 1.1 / 12
    
    # Volume spike filter: volume > 1.5 * 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # Align HTF indicators to 12h timeframe
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    R1_aligned = align_htf_to_ltf(prices, df_12h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_12h, S1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: max of EMA(34) 1d, Camarilla (need 2 bars for shift), volume MA (20)
    start_idx = max(34, 2, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(R1_aligned[i]) or
            np.isnan(S1_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        ema_34_1d_val = ema_34_1d_aligned[i]
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        vol_spike = volume_spike[i]
        r1_val = R1_aligned[i]
        s1_val = S1_aligned[i]
        
        # Trend filter: price > EMA34 (uptrend bias) or < EMA34 (downtrend bias)
        uptrend_bias = close_val > ema_34_1d_val
        downtrend_bias = close_val < ema_34_1d_val
        
        if position == 0:
            # Long: break above R1 with uptrend bias and volume spike
            long_signal = (close_val > r1_val) and \
                          uptrend_bias and \
                          vol_spike
            
            # Short: break below S1 with downtrend bias and volume spike
            short_signal = (close_val < s1_val) and \
                           downtrend_bias and \
                           vol_spike
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: close below EMA34 (trend change) or reversal below midpoint
            midpoint = (cam_high[i] + cam_low[i]) / 2  # approximate midpoint of prior bar
            if close_val < ema_34_1d_val or close_val < midpoint:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: close above EMA34 (trend change) or reversal above midpoint
            midpoint = (cam_high[i] + cam_low[i]) / 2  # approximate midpoint of prior bar
            if close_val > ema_34_1d_val or close_val > midpoint:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Camarilla_R1S1_Breakout_1dTrend_VolumeSpike_v5"
timeframe = "12h"
leverage = 1.0