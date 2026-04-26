#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter_v1
Hypothesis: 1h Camarilla R1/S1 breakout in direction of 4h EMA50 trend with 1d volume confirmation.
Camarilla R1/S1 levels provide tighter support/resistance for precise 1h entries.
4h EMA50 trend filter ensures alignment with medium-term momentum.
1d volume > 1.5x 20-period average adds conviction filter, reducing false breakouts.
Discrete sizing (0.20) limits fee drag. Session filter (08-20 UTC) reduces noise.
Target: 60-150 total trades over 4 years (15-37/year) by requiring HTF alignment, Camarilla breakout, trend alignment, volume confirmation, and session filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data ONCE before loop for HTF Camarilla and EMA
    df_4h = get_htf_data(prices, '4h')
    
    # Calculate 4h Camarilla pivot and levels (based on previous 4h bar's OHLC)
    daily_high_4h = df_4h['high'].values
    daily_low_4h = df_4h['low'].values
    daily_close_4h = df_4h['close'].values
    
    # 4h Camarilla pivot = (high + low + close) / 3
    pivot_4h = (daily_high_4h + daily_low_4h + daily_close_4h) / 3.0
    # 4h Camarilla R1 and S1
    daily_range_4h = daily_high_4h - daily_low_4h
    camarilla_4h_r1 = daily_close_4h + 1.1 * daily_range_4h / 12
    camarilla_4h_s1 = daily_close_4h - 1.1 * daily_range_4h / 12
    
    # 4h EMA50 for trend filter
    close_series_4h = pd.Series(daily_close_4h)
    ema_50_4h = close_series_4h.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (completed 4h bars only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_4h_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_4h, camarilla_4h_s1)
    ema_50_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Load 1d data ONCE before loop for volume filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 1d volume MA20
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d volume MA to 1h timeframe (completed 1d bars only)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 1h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20_1h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 50 for EMA)
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        # Session filter: only trade between 08:00 and 20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: go flat
            signals[i] = 0.0
            position = 0
            continue
        
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_ma_20_1h[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        # Volume spike conditions (both 1h and 1d must confirm)
        volume_spike_1h = volume[i] > 1.5 * vol_ma_20_1h[i]
        volume_spike_1d = volume[i] > 1.5 * vol_ma_20_aligned[i]
        volume_spike = volume_spike_1h and volume_spike_1d
        
        # Camarilla R1/S1 breakout conditions
        breakout_above = close[i] > camarilla_r1_aligned[i]  # Break above R1
        breakout_below = close[i] < camarilla_s1_aligned[i]   # Break below S1
        
        # Trend filter: price above/below 4h EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Camarilla R1 breakout with volume, in 4h uptrend, within session
            if position != 1:
                signals[i] = 0.20
                position = 1
            else:
                signals[i] = 0.20
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Camarilla S1 breakout with volume, in 4h downtrend, within session
            if position != -1:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = -0.20
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hTrend_1dVolumeFilter_v1"
timeframe = "1h"
leverage = 1.0