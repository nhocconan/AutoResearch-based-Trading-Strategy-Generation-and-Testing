#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1wTrend_v1
Hypothesis: Trade 6h Elder Ray (Bull/Bear Power) signals filtered by 1w EMA trend and volume spike.
Elder Ray measures bull/bear power relative to EMA13. In strong trends (1w EMA50), 
these signals have high win rate. Volume confirmation reduces false signals.
Designed for low trade frequency (12-37/year) to minimize fee drag on 6h timeframe.
Works in both bull and bear markets by following 1w trend.
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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # 1w EMA(50) for trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray on 6h: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume confirmation: 2.0x median volume
    vol_median = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    
    # Align HTF indicators to 6h timeframe
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Warmup: max of 1w EMA (50), EMA13 (13), volume median (30)
    start_idx = max(50, 13, 30)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(ema_13[i]) or 
            np.isnan(vol_median[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        ema_50_1w_val = ema_50_1w_aligned[i]
        ema_13_val = ema_13[i]
        bull_power_val = bull_power[i]
        bear_power_val = bear_power[i]
        close_val = close[i]
        volume_val = volume[i]
        vol_median_val = vol_median[i]
        
        if position == 0:
            # Long: bull power > 0 (strong buying), uptrend (close > 1w EMA50), volume spike
            long_signal = (bull_power_val > 0) and \
                          (close_val > ema_50_1w_val) and \
                          (volume_val > 2.0 * vol_median_val)
            # Short: bear power < 0 (strong selling), downtrend (close < 1w EMA50), volume spike
            short_signal = (bear_power_val < 0) and \
                           (close_val < ema_50_1w_val) and \
                           (volume_val > 2.0 * vol_median_val)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
                entry_price = close_val
                bars_since_entry = 0
            elif short_signal:
                signals[i] = -0.25
                position = -1
                entry_price = close_val
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long with minimum holding period
            bars_since_entry += 1
            signals[i] = 0.25
            # Exit: trend reversal (close < 1w EMA50) or bear power turns negative after minimum holding
            if bars_since_entry >= 6 and ((close_val < ema_50_1w_val) or (bear_power_val < 0)):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short with minimum holding period
            bars_since_entry += 1
            signals[i] = -0.25
            # Exit: trend reversal (close > 1w EMA50) or bull power turns positive after minimum holding
            if bars_since_entry >= 6 and ((close_val > ema_50_1w_val) or (bull_power_val > 0)):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1wTrend_v1"
timeframe = "6h"
leverage = 1.0