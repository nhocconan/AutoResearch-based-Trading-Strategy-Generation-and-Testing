#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams %R extreme + 1d EMA34 trend filter + volume spike
# Williams %R identifies overbought/oversold conditions for mean reversion entries.
# 1d EMA34 ensures alignment with daily trend to avoid fighting the trend.
# Volume spike confirms momentum behind the reversal. Designed for 50-150 total trades over 4 years (12-37/year).
# Works in bull markets via oversold bounces and in bear markets via overbought reversals.

name = "12h_WilliamsR_Extreme_1dEMA34_VolumeSpike"
timeframe = "12h"
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Williams %R (14-period) on 12h data
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.full(n, np.nan)
    # Avoid division by zero
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, -100 * (highest_high - close) / denominator, -50)
    
    # Volume confirmation: 20-period EMA on 12h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(14, n):  # Start from 14 to have valid Williams %R and volume EMA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(williams_r[i]) or np.isnan(vol_ema_20[i]) or 
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: Williams %R oversold (< -80) in uptrend alignment with volume spike
            if williams_r[i] < -80 and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R overbought (> -20) in downtrend alignment with volume spike
            elif williams_r[i] > -20 and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R rises above -50 or loses uptrend alignment
            if williams_r[i] > -50 or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R falls below -50 or loses downtrend alignment
            if williams_r[i] < -50 or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals