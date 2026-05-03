#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
# Uses 12h timeframe as primary (target: 50-150 total trades over 4 years, 12-37/year).
# Camarilla pivot levels provide high-probability reversal/breakout levels derived from prior 1d range.
# 1d EMA34 ensures alignment with daily trend to avoid counter-trend trades.
# Volume confirmation filters false breakouts. Designed for low trade frequency to minimize fee drag.
# Works in bull markets via upward breaks above R3 and in bear markets via downward breaks below S3.

name = "12h_Camarilla_R3S3_1dEMA34_VolumeSpike"
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
    
    # Get 1d data for Camarilla pivot calculation and EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from prior 1d bar (H1, L1, C1)
    # R3 = C1 + (H1 - L1) * 1.1/2
    # S3 = C1 - (H1 - L1) * 1.1/2
    camarilla_R3 = np.full(n, np.nan)
    camarilla_S3 = np.full(n, np.nan)
    
    for i in range(len(df_1d)):
        # Get prior completed 1d bar values
        h1 = df_1d['high'].iloc[i]
        l1 = df_1d['low'].iloc[i]
        c1 = df_1d['close'].iloc[i]
        r3 = c1 + (h1 - l1) * 1.1 / 2
        s3 = c1 - (h1 - l1) * 1.1 / 2
        
        # Map to 12h bars that fall within this 1d bar
        # The 1d bar at index i corresponds to 12h bars from i*2 to i*2+1 (since 2x 12h = 1d)
        start_idx = i * 2
        end_idx = start_idx + 2
        if end_idx <= n:
            camarilla_R3[start_idx:end_idx] = r3
            camarilla_S3[start_idx:end_idx] = s3
    
    # Volume confirmation: 20-period EMA on 12h
    vol_ema_20 = np.full(n, np.nan)
    vol_series = pd.Series(volume)
    vol_ema_20_values = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_20[:] = vol_ema_20_values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(2, n):  # Start from 2 to have prior 1d bar for Camarilla
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(camarilla_R3[i]) or np.isnan(camarilla_S3[i]) or 
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R3 in uptrend alignment with volume spike
            if close[i] > camarilla_R3[i] and ema_34_1d_aligned[i] < close[i] and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Camarilla S3 in downtrend alignment with volume spike
            elif close[i] < camarilla_S3[i] and ema_34_1d_aligned[i] > close[i] and volume_spike:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Camarilla S3 or loses uptrend alignment
            if close[i] < camarilla_S3[i] or ema_34_1d_aligned[i] >= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Camarilla R3 or loses downtrend alignment
            if close[i] > camarilla_R3[i] or ema_34_1d_aligned[i] <= close[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals