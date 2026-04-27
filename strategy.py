#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses daily Camarilla pivot R1/S1 levels for breakout entries in the direction of 1d EMA34 trend.
Enter long when price breaks above R1 AND 1d close > EMA34 (uptrend) AND volume > 2.0x 20-period average.
Enter short when price breaks below S1 AND 1d close < EMA34 (downtrend) AND volume > 2.0x 20-period average.
Exit when price returns to the pivot level (PP) or trend reverses.
Designed for 4h timeframe to achieve 75-200 total trades over 4 years.
Camarilla levels provide institutional support/resistance; volume spike confirms institutional participation.
Works in both bull and bear markets by following 1d trend while using Camarilla for precise breakout entries.
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
    
    # Get 1d data for Camarilla pivots and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot levels
    # PP = (H + L + C) / 3
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    pp = typical_price.values
    r1 = df_1d['close'].values + (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    s1 = df_1d['close'].values - (df_1d['high'].values - df_1d['low'].values) * 1.1 / 12
    
    # Align Camarilla levels to 4h timeframe (wait for daily close)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 1d EMA34 (34), volume avg (20), and daily data
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: breakout of Camarilla R1/S1 with 1d trend filter AND volume spike
            # Long: price breaks above R1 AND 1d uptrend AND volume confirmation
            long_condition = (close_val > r1_val) and (close_val > ema_val) and vol_conf
            # Short: price breaks below S1 AND 1d downtrend AND volume confirmation
            short_condition = (close_val < s1_val) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when price returns to pivot point OR trend breaks
            exit_condition = (close_val <= pp_val) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when price returns to pivot point OR trend breaks
            exit_condition = (close_val >= pp_val) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0