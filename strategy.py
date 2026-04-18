#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index (Bull/Bear Power) with daily EMA34 filter and volume spike
# Elder Ray measures bull power (high - EMA) and bear power (low - EMA) to show trend strength.
# Daily EMA34 filter ensures we trade only in the direction of higher timeframe trend.
# Volume spike confirms momentum behind the move.
# Designed for 6h timeframe with low trade frequency (12-37/year) to minimize fee drift.
# Works in bull markets (bull power positive, price above EMA) and bear markets (bear power negative, price below EMA).

name = "6h_ElderRay_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for EMA34 filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(13) for 6h timeframe (Elder Ray uses typically 13-period EMA)
    ema_period = 13
    ema_13 = pd.Series(close).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    
    # Calculate Bull Power and Bear Power
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Calculate daily EMA(34) for trend filter
    close_d = df_1d['close'].values
    ema_34_d = pd.Series(close_d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_d)
    
    # Calculate volume spike detector (volume > 1.5 * 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, ema_period)  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_13[i]) or np.isnan(ema_34_d_aligned[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Bull Power positive AND price above daily EMA34 AND volume spike
            long_condition = (bull_power[i] > 0) and (close[i] > ema_34_d_aligned[i]) and vol_spike[i]
            # Short: Bear Power negative AND price below daily EMA34 AND volume spike
            short_condition = (bear_power[i] < 0) and (close[i] < ema_34_d_aligned[i]) and vol_spike[i]
            
            if long_condition:
                signals[i] = 0.25
                position = 1
            elif short_condition:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Bull Power turns negative OR price crosses below daily EMA34
            exit_condition = (bull_power[i] <= 0) or (close[i] < ema_34_d_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Bear Power turns positive OR price crosses above daily EMA34
            exit_condition = (bear_power[i] >= 0) or (close[i] > ema_34_d_aligned[i])
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals