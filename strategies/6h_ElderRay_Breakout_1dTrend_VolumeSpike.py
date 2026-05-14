#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 1d Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) to measure buying/selling pressure. In uptrend (price > EMA34), go long when Bull Power > 0 and volume spikes; in downtrend (price < EMA34), go short when Bear Power < 0 and volume spikes. Exit when Elder Power reverses or price crosses EMA34. Volume confirmation (>2x average) ensures conviction. 6h timeframe targets 50-150 trades over 4 years (12-37/year). Works in bull markets via buying pressure and in bear markets via selling pressure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d EMA13 for Elder Ray
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d Elder Ray components
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    bull_power_1d = high_1d - ema_13_1d  # Buying pressure
    bear_power_1d = low_1d - ema_13_1d   # Selling pressure
    
    # Align all 1d indicators to 6h timeframe
    ema_13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need EMA34 (34), EMA13 (13), volume avg (20)
    start_idx = max(34, 13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(bull_power_1d_aligned[i]) or 
            np.isnan(bear_power_1d_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        bull_power = bull_power_1d_aligned[i]
        bear_power = bear_power_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Determine trend: price > EMA34 = uptrend, price < EMA34 = downtrend
            is_uptrend = close_val > ema_1d_val
            is_downtrend = close_val < ema_1d_val
            
            if is_uptrend:
                # Uptrend: long when buying pressure exists and volume confirms
                if (bull_power > 0) and vol_conf:
                    signals[i] = size
                    position = 1
            elif is_downtrend:
                # Downtrend: short when selling pressure exists and volume confirms
                if (bear_power < 0) and vol_conf:
                    signals[i] = -size
                    position = -1
        elif position == 1:
            # Exit long: buying pressure fades or trend changes
            exit_condition = (bull_power <= 0) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: selling pressure fades or trend changes
            exit_condition = (bear_power >= 0) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0