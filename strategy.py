#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_VolumeSpike
Hypothesis: Daily Donchian(20) breakouts with weekly trend filter and volume spike capture strong momentum moves.
In bull markets: price breaks above upper Donchian(20) with weekly uptrend and volume spike → long.
In bear markets: price breaks below lower Donchian(20) with weekly downtrend and volume spike → short.
Uses weekly EMA50 for trend and volume > 2.0x 20-day median for confirmation. Target: 30-100 trades over 4 years.
Donchian channels provide clear structure, weekly filter avoids counter-trend trades, volume spike confirms conviction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:  # Need 20 for Donchian and volume median
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20 for Donchian, 50 for weekly EMA)
    start_idx = 50
    
    for i in range(start_idx, n):
        close_val = close[i]
        upper_val = upper[i]
        lower_val = lower[i]
        ema_val = ema_50_1w_aligned[i]
        
        # Skip if any data not ready
        if (np.isnan(upper_val) or 
            np.isnan(lower_val) or 
            np.isnan(ema_val) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper Donchian with volume spike and weekly uptrend
        long_condition = (close_val > upper_val) and volume_spike[i] and (close_val > ema_val)
        # Short logic: price breaks below lower Donchian with volume spike and weekly downtrend
        short_condition = (close_val < lower_val) and volume_spike[i] and (close_val < ema_val)
        
        # Exit logic: trend reversal (price crosses weekly EMA50)
        exit_long = close_val < ema_val
        exit_short = close_val > ema_val
        
        if long_condition and position != 1:
            signals[i] = base_size
            position = 1
        elif short_condition and position != -1:
            signals[i] = -base_size
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
    
    return signals

name = "1d_Donchian20_Breakout_1wTrend_VolumeSpike"
timeframe = "1d"
leverage = 1.0