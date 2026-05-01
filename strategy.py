#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R reversal with 1d EMA50 trend filter and volume spike confirmation.
# Uses Williams %R(14) from 1d for reversal signals (oversold < -80 for long, overbought > -20 for short),
# filtered by 1d EMA50 trend and volume > 2.0x 20-period median. Works in bull (buy oversold with uptrend) 
# and bear (sell overbought with downtrend). Discrete position sizing (0.25) to minimize fee churn.
# Target: 50-150 total trades over 4 years.

name = "12h_WilliamsR_Reversal_1dEMA50_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Williams %R(14)
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    close_1d = df_1d['close'].values
    williams_r = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 12h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50, Williams %R, and volume median
    start_idx = max(50, 14, 20) + 1  # 51
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(williams_r_aligned[i]) or
            np.isnan(vol_median_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA50 direction
        uptrend = curr_close > ema_50_1d_aligned[i]
        downtrend = curr_close < ema_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Williams %R reversal conditions
        oversold = williams_r_aligned[i] < -80   # Oversold condition for long
        overbought = williams_r_aligned[i] > -20  # Overbought condition for short
        
        if position == 0:  # Flat - look for new entries
            # Long: Oversold AND uptrend AND volume confirmation
            if oversold and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Overbought AND downtrend AND volume confirmation
            elif overbought and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit when Williams %R rises above -50 (momentum fading)
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading)
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals