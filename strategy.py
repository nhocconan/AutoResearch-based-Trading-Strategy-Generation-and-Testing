#!/usr/bin/env python3
"""
6h_ElderRay_Breakout_1dTrend_VolumeSpike
Hypothesis: Uses 6h Elder Ray (Bull/Bear Power) breakouts filtered by 1d EMA34 trend and volume spike (>2x average).
Enter long when 6h Bull Power > 0 AND price breaks above 6h EMA20 AND 1d close > 1d EMA34 (uptrend) AND volume > 2x average.
Enter short when 6h Bear Power < 0 AND price breaks below 6h EMA20 AND 1d close < 1d EMA34 (downtrend) AND volume > 2x average.
Exit when Elder Power reverses sign OR 1d trend breaks.
Designed for 6h timeframe with moderate entries to avoid fee drag: target 12-37 trades/year.
Works in both bull and bear markets via 1d trend filter and volume confirmation to avoid false signals.
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
    
    # Get 6h data for Elder Ray calculation and EMA20
    df_6h = get_htf_data(prices, '6h')
    
    # Calculate EMA20 on 6h close
    close_6h_series = pd.Series(df_6h['close'].values)
    ema_20_6h = close_6h_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    close_6h_for_ema = df_6h['close'].values
    ema_13_6h = pd.Series(close_6h_for_ema).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_6h['high'].values - ema_13_6h
    bear_power = df_6h['low'].values - ema_13_6h
    
    # Align 6h indicators to 6h timeframe (identity alignment)
    ema_20_6h_aligned = align_htf_to_ltf(prices, df_6h, ema_20_6h)
    bull_power_aligned = align_htf_to_ltf(prices, df_6h, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_6h, bear_power)
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA34 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 2 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need 6h EMA13 (13), EMA20 (20), 1d EMA34 (34), volume avg (20)
    start_idx = max(13, 20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(ema_20_6h_aligned[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        ema_1d_val = ema_34_1d_aligned[i]
        ema_20_val = ema_20_6h_aligned[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Look for entry: Elder Power alignment with price > EMA20 for long, < EMA20 for short
            # Long: Bull Power > 0 AND price > EMA20 AND 1d uptrend AND volume
            long_condition = (bull_power_val > 0) and (close_val > ema_20_val) and (close_val > ema_1d_val) and vol_conf
            # Short: Bear Power < 0 AND price < EMA20 AND 1d downtrend AND volume
            short_condition = (bear_power_val < 0) and (close_val < ema_20_val) and (close_val < ema_1d_val) and vol_conf
            
            if long_condition:
                signals[i] = size
                position = 1
            elif short_condition:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long when Bull Power <= 0 OR price < EMA20 OR 1d trend breaks
            exit_condition = (bull_power_val <= 0) or (close_val < ema_20_val) or (close_val < ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short when Bear Power >= 0 OR price > EMA20 OR 1d trend breaks
            exit_condition = (bear_power_val >= 0) or (close_val > ema_20_val) or (close_val > ema_1d_val)
            
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_Breakout_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0