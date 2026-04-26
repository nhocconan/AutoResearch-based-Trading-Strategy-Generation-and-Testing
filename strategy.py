#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1
Hypothesis: 12h Camarilla R1/S1 breakout in direction of 1d EMA34 trend with volume confirmation.
12h timeframe targets 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
Camarilla R1/S1 levels provide meaningful support/resistance with fewer false breaks than tighter levels.
1d EMA34 trend filter ensures alignment with higher timeframe momentum.
Volume confirmation adds conviction filter. Discrete sizing (0.25) limits fee changes.
Designed to work in both bull (breakouts with trend) and bear (mean reversion via failed breaks) markets.
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
    
    # Load 1d data ONCE before loop for HTF Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily Camarilla pivot and levels (based on previous day's OHLC)
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Camarilla pivot = (daily_high + daily_low + daily_close) / 3
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    # Daily Camarilla R1 and S1
    daily_range = daily_high - daily_low
    camarilla_d_r1 = daily_close + 1.1 * daily_range / 12
    camarilla_d_s1 = daily_close - 1.1 * daily_range / 12
    
    # Daily EMA34 for trend filter
    close_series_1d = pd.Series(daily_close)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 12h timeframe (completed daily bars only)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_d_s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # 12h volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA and 34 for EMA)
    start_idx = max(20, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume spike condition
        volume_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Camarilla R1/S1 breakout conditions
        breakout_above = close[i] > camarilla_r1_aligned[i]  # Break above R1
        breakout_below = close[i] < camarilla_s1_aligned[i]   # Break below S1
        
        # Trend filter: price above/below daily EMA34
        uptrend = close[i] > ema_34_aligned[i]
        downtrend = close[i] < ema_34_aligned[i]
        
        if breakout_above and volume_spike and uptrend:
            # Long signal: Camarilla R1 breakout with volume, in daily uptrend
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        elif breakout_below and volume_spike and downtrend:
            # Short signal: Camarilla S1 breakout with volume, in daily downtrend
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1dTrend_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0