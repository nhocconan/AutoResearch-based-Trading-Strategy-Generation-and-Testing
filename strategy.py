#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R extreme reversal with 1d EMA34 trend filter and volume confirmation
# Williams %R identifies overbought/oversold conditions; reversals from extreme levels (>80 or <20)
# with 1d trend alignment and volume spike provide high-probability mean reversion trades.
# Designed for low trade frequency (12-37/year) on 6h timeframe to minimize fee drag.
# Works in both bull and bear markets by trading reversals in the direction of the higher timeframe trend.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R, EMA34, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - df_1d['close'].values) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 1d volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20 = pd.Series(df_1d['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = df_1d['volume'].values > (2.0 * vol_ema_20)
    
    # Align 1d indicators to 6h timeframe
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 from below (oversold reversal) with volume spike in uptrend
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema_34_aligned[i] and volume_spike_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 from above (overbought reversal) with volume spike in downtrend
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema_34_aligned[i] and volume_spike_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses below -50 or re-enters overbought territory
            if williams_r_aligned[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses above -50 or re-enters oversold territory
            if williams_r_aligned[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals