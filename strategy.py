#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 1d Camarilla pivot levels (R1/S1) for breakout entries with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above R1 AND 1d close > EMA34 (uptrend) AND volume > 2.0 * 20-period average.
Short when price breaks below S1 AND 1d close < EMA34 (downtrend) AND volume > 2.0 * 20-period average.
Exit when price returns to the pivot level (R1 for longs, S1 for shorts) OR trend reverses.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years with low fee drag.
Works in both bull and bear markets by following 1d trend while using Camarilla levels for precise breakout entries.
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
    
    # Get 1d data for trend filter and Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Camarilla pivot levels: R1, S1
    # Camarilla formulas: R1 = close + 1.1*(high-low)*1.1/12, S1 = close - 1.1*(high-low)*1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_r1 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        high_val = high[i]
        low_val = low[i]
        ema_val = ema_34_aligned[i]
        r1_level = camarilla_r1_aligned[i]
        s1_level = camarilla_s1_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 1d trend filter AND volume
            # Long: price breaks above R1 (minor resistance) AND 1d uptrend AND volume
            long_condition = (close_val > r1_level) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S1 (minor support) AND 1d downtrend AND volume
            short_condition = (close_val < s1_level) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to R1 level OR trend breaks
            exit_condition = (close_val <= r1_level) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to S1 level OR trend breaks
            exit_condition = (close_val >= s1_level) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0