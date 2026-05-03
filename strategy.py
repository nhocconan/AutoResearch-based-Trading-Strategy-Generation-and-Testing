#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R Extreme + 1d EMA34 Trend + Volume Spike
# Williams %R identifies overbought/oversold conditions; extreme readings (<-90 or >-10) 
# often precede reversals. Combined with 1d EMA34 trend filter to ensure we trade 
# with the daily trend, and volume spike to confirm institutional participation.
# Designed for low trade frequency (target: 12-37/year) to minimize fee drag on 6h timeframe.
# Works in both bull and bear markets by trading with the higher timeframe trend.

name = "6h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Williams %R, EMA, and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high - df_1d['close'].values) / (highest_high - lowest_low) * -100
    # Avoid division by zero
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
    
    for i in range(40, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(volume_spike_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine daily trend direction
        is_uptrend = close[i] > ema_34_aligned[i]
        is_downtrend = close[i] < ema_34_aligned[i]
        
        if position == 0:
            # Long: Williams %R extreme oversold (<-90) in uptrend with volume spike
            if williams_r_aligned[i] < -90 and is_uptrend and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R extreme overbought (>-10) in downtrend with volume spike
            elif williams_r_aligned[i] > -10 and is_downtrend and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R returns above -50 (momentum fading) or hits extreme overbought
            if williams_r_aligned[i] > -50 or williams_r_aligned[i] > -10:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R returns below -50 (momentum fading) or hits extreme oversold
            if williams_r_aligned[i] < -50 or williams_r_aligned[i] < -90:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals