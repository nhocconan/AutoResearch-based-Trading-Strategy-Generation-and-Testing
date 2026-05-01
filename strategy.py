#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme with 1d EMA34 trend filter and volume spike confirmation.
# Williams %R identifies overbought/oversold conditions; extreme readings (< -90 or > -10) 
# signal potential reversals when aligned with 1d trend and confirmed by volume spikes.
# Works in bull markets (buy oversold dips in uptrend) and bear markets (sell overbought rallies in downtrend).
# Discrete position sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeConfirm_v1"
timeframe = "6h"
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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Williams %R on 6h data (14-period)
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    rr = highest_high_14 - lowest_low_14
    williams_r = np.where(rr != 0, -100 * (highest_high_14 - close) / rr, -50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for Williams %R and EMA34
    start_idx = 34
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(williams_r[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation: current volume > 2.0x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 2.0)
        
        # Williams %R extreme conditions
        oversold = williams_r[i] < -90   # Extremely oversold
        overbought = williams_r[i] > -10  # Extremely overbought
        
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
            if williams_r[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit when Williams %R falls below -50 (momentum fading)
            if williams_r[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals