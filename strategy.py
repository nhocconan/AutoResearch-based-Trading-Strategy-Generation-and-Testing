#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeConfirm
Hypothesis: Uses 6h Elder Ray (Bull/Bear Power) breakout with 1d EMA trend filter and volume confirmation.
Long when Bull Power crosses above zero AND 1d close > EMA34 (uptrend) AND volume > 1.5 * 20-period average.
Short when Bear Power crosses below zero AND 1d close < EMA34 (downtrend) AND volume > 1.5 * 20-period average.
Exit when Elder Power reverses sign OR trend breaks.
Designed for 6h timeframe to achieve 50-150 total trades over 4 years with low fee drag.
Elder Ray measures buying/selling pressure relative to EMA13, effective in both bull and bear markets.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 6h Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Need EMA13 on 6h data
    close_s = pd.Series(close)
    ema_13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    bull_power = high - ema_13  # Buying pressure
    bear_power = low - ema_13   # Selling pressure (negative values)
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 6h EMA13 (13), 1d EMA34 (34), volume avg (20)
    start_idx = max(13, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        ema_val = ema_34_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Elder Power crosses zero with 1d trend filter AND volume
            # Long: Bull Power crosses above zero (buying pressure) AND 1d uptrend AND volume
            long_cross = (bull_val > 0) and (bull_power[i-1] <= 0) if i > 0 else False
            long_condition = long_cross and (close[i] > ema_val) and vol_conf
            # Short: Bear Power crosses below zero (selling pressure) AND 1d downtrend AND volume
            short_cross = (bear_val < 0) and (bear_power[i-1] >= 0) if i > 0 else False
            short_condition = short_cross and (close[i] < ema_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Bull Power turns negative OR trend breaks
            exit_condition = (bull_val <= 0) or (close[i] < ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Bear Power turns positive OR trend breaks
            exit_condition = (bear_val >= 0) or (close[i] > ema_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0