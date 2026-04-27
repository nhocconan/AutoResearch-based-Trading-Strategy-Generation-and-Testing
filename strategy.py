#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeSpike
Hypothesis: Uses 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure buying/selling pressure.
Enter long when Bull Power > 0 AND 1d close > EMA34 (uptrend) AND volume spike.
Enter short when Bear Power < 0 AND 1d close < EMA34 (downtrend) AND volume spike.
Exit when power reverses or volume dries up. Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year).
Works in both bull and bear markets by following 1d trend while using Elder Ray for momentum confirmation.
Volume spike filter reduces false signals. Discrete position sizing (0.25) minimizes fee drag.
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
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray power calculations
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate Elder Ray components: Bull Power = High - EMA13, Bear Power = Low - EMA13
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power = high_1d - ema_13_1d  # Measures buying pressure above average
    bear_power = low_1d - ema_13_1d   # Measures selling pressure below average (negative when strong)
    
    # Align 1d indicators to 6h timeframe (completed bars only)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital (discrete level)
    
    # Warmup: need 1d EMA34 (34) and EMA13 (13) + volume avg (20)
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_val = ema_34_aligned[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Elder Ray power with 1d EMA34 trend filter AND volume spike
            # Long: Bull Power > 0 (buying pressure) AND price above EMA34 (1d uptrend) AND volume spike
            long_condition = (bull_val > 0) and (close_val > ema_val) and vol_conf
            # Short: Bear Power < 0 (selling pressure) AND price below EMA34 (1d downtrend) AND volume spike
            short_condition = (bear_val < 0) and (close_val < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Bull Power turns negative (buying pressure fades) OR trend breaks
            exit_condition = (bull_val <= 0) or (close_val < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Bear Power turns positive (selling pressure fades) OR trend breaks
            exit_condition = (bear_val >= 0) or (close_val > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0