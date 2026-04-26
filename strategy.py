#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_1dTrend
Hypothesis: Donchian(20) breakout with 1d EMA34 trend filter and volume spike confirmation.
In bull markets: price breaks above upper band with 1d uptrend → long.
In bear markets: price breaks below lower band with 1d downtrend → short.
Volume spike (>2x median) confirms breakout strength.
Uses discrete sizing (0.25) to minimize fee drag.
Target: 75-200 trades over 4 years (19-50/year).
Works in both regimes by requiring alignment with daily trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    period = 20
    max_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    min_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Volume confirmation: volume > 2.0x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_spike = volume > (vol_median * 2.0)
    
    # Load 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Daily EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(max_high[i]) or np.isnan(min_low[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: price breaks above upper band with volume spike and daily uptrend
        long_condition = close[i] > max_high[i] and volume_spike[i] and (close[i] > ema_34_1d_aligned[i])
        # Short logic: price breaks below lower band with volume spike and daily downtrend
        short_condition = close[i] < min_low[i] and volume_spike[i] and (close[i] < ema_34_1d_aligned[i])
        
        # Exit logic: price re-enters the channel or daily trend reversal
        exit_long = close[i] < min_low[i] or close[i] < ema_34_1d_aligned[i]
        exit_short = close[i] > max_high[i] or close[i] > ema_34_1d_aligned[i]
        
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

name = "4h_Donchian20_Breakout_VolumeSpike_1dTrend"
timeframe = "4h"
leverage = 1.0