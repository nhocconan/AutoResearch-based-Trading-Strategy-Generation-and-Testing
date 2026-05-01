#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Williams %R extreme + 1d EMA50 trend filter + volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (< -80 or > -20) signal potential reversals.
# In bull markets, buy when %R < -80 (oversold) with uptrend and volume confirmation.
# In bear markets, sell when %R > -20 (overbought) with downtrend and volume confirmation.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_WilliamsR_Extreme_1dEMA50_VolumeConfirm_v1"
timeframe = "4h"
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
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R (14-period) on 1d timeframe
    # %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(df_1d['high']).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(df_1d['low']).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - df_1d['close'].values) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Align Williams %R to 4h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for EMA50 and Williams %R
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(williams_r_aligned[i])):
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
        
        # Williams %R extreme conditions
        williams_r_val = williams_r_aligned[i]
        oversold = williams_r_val < -80.0   # Oversold condition
        overbought = williams_r_val > -20.0 # Overbought condition
        
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
            # Exit when Williams %R returns from oversold (above -50) or overbought
            if williams_r_val > -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R returns from overbought (below -50) or oversold
            if williams_r_val < -50.0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals