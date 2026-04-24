#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme with 1d Elder Ray Power confirmation and volume spike.
- Primary timeframe: 6h targeting 50-150 total trades over 4 years (12-37/year).
- HTF: 1d Elder Ray Power (bull/bear power) for trend direction and 1w Williams %R for extreme conditions.
- Entry: Long when 6h Williams %R < -80 (oversold) AND 1d Elder Bull Power > 0 AND 1w Williams %R < -80 AND volume > 1.5 * volume MA(24).
         Short when 6h Williams %R > -20 (overbought) AND 1d Elder Bear Power < 0 AND 1w Williams %R > -20 AND volume > 1.5 * volume MA(24).
- Exit: Close-based reversal - exit long when 6h Williams %R > -50, exit short when 6h Williams %R < -50.
- Signal size: 0.25 discrete to balance profit potential and drawdown control.
Designed to work in both bull and bear markets via extreme readings aligned with higher timeframe trend and momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Extract price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Elder Ray Power (EMA13-based)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d Elder Ray Power: Bull Power = High - EMA13, Bear Power = Low - EMA13
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    ema_13 = pd.Series(df_1d_close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = df_1d_high - ema_13
    bear_power = df_1d_low - ema_13
    
    # Get 1w data for Williams %R (14-period)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate 1w Williams %R: (Highest High - Close) / (Highest High - Lowest Low) * -100
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    highest_high_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_low_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    wr_1w = (highest_high_1w - close_1w) / (highest_high_1w - lowest_low_1w) * -100
    # Handle division by zero (when high==low)
    wr_1w = np.where((highest_high_1w - lowest_low_1w) == 0, -50, wr_1w)
    
    # Align HTF indicators to 6h
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    wr_1w_aligned = align_htf_to_ltf(prices, df_1w, wr_1w)
    
    # Calculate 6h Williams %R (14-period)
    highest_high_6h = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_6h = pd.Series(low).rolling(window=14, min_periods=14).min().values
    wr_6h = (highest_high_6h - close) / (highest_high_6h - lowest_low_6h) * -100
    wr_6h = np.where((highest_high_6h - lowest_low_6h) == 0, -50, wr_6h)
    
    # Calculate volume MA(24) for confirmation (using 6h data)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(100, 14, 24)  # Need enough bars for Williams %R and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(wr_6h[i]) or np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(wr_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_wr_6h = wr_6h[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Check for entry signals with volume confirmation (1.5x threshold)
            vol_confirmed = curr_volume > 1.5 * vol_ma[i]
            
            # Long: 6h Williams %R < -80 (oversold) AND 1d Bull Power > 0 AND 1w Williams %R < -80 AND volume confirmed
            if curr_wr_6h < -80 and bull_power_aligned[i] > 0 and wr_1w_aligned[i] < -80 and vol_confirmed:
                signals[i] = 0.25
                position = 1
            # Short: 6h Williams %R > -20 (overbought) AND 1d Bear Power < 0 AND 1w Williams %R > -20 AND volume confirmed
            elif curr_wr_6h > -20 and bear_power_aligned[i] < 0 and wr_1w_aligned[i] > -20 and vol_confirmed:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when 6h Williams %R > -50 (recovering from oversold)
            if curr_wr_6h > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when 6h Williams %R < -50 (declining from overbought)
            if curr_wr_6h < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_1dElderRay_Power_1wWvR_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0