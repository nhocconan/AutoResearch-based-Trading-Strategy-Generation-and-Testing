#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray power filter and volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 1d Elder Bull Power > 0 AND 6h volume > 1.5x 20-period MA.
Short when Williams %R(14) crosses below -20 (overbought) AND 1d Elder Bear Power < 0 AND 6h volume > 1.5x 20-period MA.
Exit when Williams %R crosses above -20 (for long) or below -80 (for short) or Elder Ray power reverses.
Uses 1d HTF for Elder Ray trend filter to align with daily momentum, volume spike for confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R captures mean reversions in ranging markets, Elder Ray filters trend direction.
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
    
    # Calculate Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Calculate 1d Elder Ray Power (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:  # EMA13 needs min_periods
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 of close
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Bull Power = High - EMA13
    bull_power = high_1d - ema_13_1d
    # Elder Bear Power = Low - EMA13
    bear_power = low_1d - ema_13_1d
    
    # Align Elder Ray to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 6h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 20)  # Williams %R, Elder Ray, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        wr = williams_r[i]
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_20[i]
        
        # Calculate Williams %R crossover signals (using previous bar to avoid look-ahead)
        wr_prev = williams_r[i-1] if i > 0 else -50
        wr_cross_above_80 = wr_prev <= -80 and wr > -80  # Oversold exit
        wr_cross_below_20 = wr_prev >= -20 and wr < -20   # Overbought exit
        wr_cross_above_20 = wr_prev <= -20 and wr > -20   # Oversold entry
        wr_cross_below_80 = wr_prev >= -80 and wr < -80   # Overbought entry
        
        # Volume filter: 6h volume > 1.5x 20-period MA
        vol_filter = volume[i] > 1.5 * vol_ma_val
        
        if position == 0:
            # Long: Williams %R crosses above -80 (exit oversold) AND Bull Power > 0 AND volume filter
            if wr_cross_above_80 and bull_power_val > 0 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (exit overbought) AND Bear Power < 0 AND volume filter
            elif wr_cross_below_20 and bear_power_val < 0 and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: Williams %R crosses above -20 (re-enter overbought) OR Bull Power <= 0
                if wr_cross_above_20 or bull_power_val <= 0:
                    exit_signal = True
            elif position == -1:
                # Short exit: Williams %R crosses below -80 (re-enter oversold) OR Bear Power >= 0
                if wr_cross_below_80 or bear_power_val >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_Power_VolumeFilter"
timeframe = "6h"
leverage = 1.0