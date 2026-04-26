#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter
Hypothesis: Camarilla pivot R1/S1 breakout on 1h with 4h EMA50 trend filter and 1d volume confirmation (>1.5x 20-period MA). 
Long when price breaks above R1 in 4h uptrend with daily volume spike. Short when price breaks below S1 in 4h downtrend with volume spike.
Uses discrete position sizing (0.20) to minimize fee churn. Target: 15-37 trades/year (60-150 total over 4 years).
Uses 4h for signal direction, 1h only for entry timing. Session filter (08-20 UTC) to reduce noise.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation and trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 4h OHLC
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Shift by 1 to use prior 4h bar's OHLC for current bar's levels
    close_4h_prev = np.roll(close_4h, 1)
    high_4h_prev = np.roll(high_4h, 1)
    low_4h_prev = np.roll(low_4h, 1)
    close_4h_prev[0] = np.nan
    high_4h_prev[0] = np.nan
    low_4h_prev[0] = np.nan
    
    # Camarilla R1, S1, R3, S3 levels
    camarilla_range = high_4h_prev - low_4h_prev
    r1 = close_4h_prev + camarilla_range * 1.1 / 12
    s1 = close_4h_prev - camarilla_range * 1.1 / 12
    r3 = close_4h_prev + camarilla_range * 1.1 / 4
    s3 = close_4h_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # 4h EMA50 trend filter
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    uptrend_4h = close > ema_50_4h_aligned
    downtrend_4h = close < ema_50_4h_aligned
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 1d volume > 1.5x 20-period MA
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = vol_1d > (vol_ma_1d * 1.5)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 4h EMA + 20 for 1d volume MA + 1 for Camarilla shift)
    start_idx = 71
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            # Hold current position outside session
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Skip if any data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_1d_aligned[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        if position == 0:
            # Long: price breaks above R1 with 4h uptrend and 1d volume spike
            if (close[i] > r1_aligned[i] and 
                uptrend_4h[i] and volume_spike_1d_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 with 4h downtrend and 1d volume spike
            elif (close[i] < s1_aligned[i] and 
                  downtrend_4h[i] and volume_spike_1d_aligned[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below R3 (strong reversal) OR 4h trend changes to downtrend
            if (close[i] < r3_aligned[i] or not uptrend_4h[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above S3 (strong reversal) OR 4h trend changes to uptrend
            if (close[i] > s3_aligned[i] or not downtrend_4h[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter"
timeframe = "1h"
leverage = 1.0