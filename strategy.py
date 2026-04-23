#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R reversal with 1d Elder Ray (Bull/Bear Power) filter and 1w volume spike confirmation.
Long when Williams %R(14) crosses above -80 (oversold) AND 1d Bull Power > 0 AND 1w volume > 1.5x 20-period average.
Short when Williams %R(14) crosses below -20 (overbought) AND 1d Bear Power < 0 AND 1w volume > 1.5x 20-period average.
Exit when Williams %R crosses back through -50 (mean reversion) or 1w volume drops below average.
Uses 1d HTF for Elder Ray trend and 1w for volume filter to avoid low-quality reversals in ranging markets.
Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Williams %R formula: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
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
    
    # Calculate Williams %R(14) on primary timeframe
    if n < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high - lowest_low) != 0, 
                          ((highest_high - close) / (highest_high - lowest_low)) * -100, 
                          -50)
    
    # Calculate 1d Elder Ray (Bull/Bear Power) for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d  # Bull Power = High - EMA13
    bear_power = low_1d - ema_13_1d   # Bear Power = Low - EMA13
    
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Calculate 1w volume average for spike filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    volume_1w = df_1w['volume'].values
    vol_ma_1w = pd.Series(volume_1w).rolling(window=20, min_periods=20).mean().values
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 13, 20)  # Williams %R(14), EMA13, volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(bull_power_aligned[i]) or 
            np.isnan(bear_power_aligned[i]) or np.isnan(vol_ma_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        wr = williams_r[i]
        bull_val = bull_power_aligned[i]
        bear_val = bear_power_aligned[i]
        vol_ma_val = vol_ma_1w_aligned[i]
        
        # Calculate Williams %R crossover signals
        if i >= start_idx + 1:
            wr_prev = williams_r[i-1]
            # Long: %R crosses above -80 (oversold reversal)
            wr_long_signal = wr_prev <= -80 and wr > -80
            # Short: %R crosses below -20 (overbought reversal)
            wr_short_signal = wr_prev >= -20 and wr < -20
            # Exit: %R crosses -50 (mean reversion)
            wr_exit_long = wr_prev > -50 and wr <= -50  # Long exit
            wr_exit_short = wr_prev < -50 and wr >= -50  # Short exit
        else:
            wr_long_signal = False
            wr_short_signal = False
            wr_exit_long = False
            wr_exit_short = False
        
        if position == 0:
            # Long: %R crosses above -80 AND Bull Power > 0 AND 1w volume spike
            if wr_long_signal and bull_val > 0 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # Short: %R crosses below -20 AND Bear Power < 0 AND 1w volume spike
            elif wr_short_signal and bear_val < 0 and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: %R crosses -50 (mean reversion) OR 1w volume drops below average
                if wr_exit_long or volume[i] < vol_ma_val:
                    exit_signal = True
            elif position == -1:
                # Short exit: %R crosses -50 (mean reversion) OR 1w volume drops below average
                if wr_exit_short or volume[i] < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Reversal_1dElderRay_1wVolumeSpike"
timeframe = "6h"
leverage = 1.0