#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeConfirm
Hypothesis: Daily Donchian(20) breakout with weekly EMA50 trend filter and volume confirmation.
Works in bull markets by catching breakouts with trend alignment. Works in bear markets by
only taking shorts when price breaks below Donchian low in a downtrend (weekly EMA50).
Volume confirmation reduces false breakouts. Discrete sizing (0.25) minimizes fee drag.
Target: 30-100 trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # need 20 for Donchian + 34 for warmup safety
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period median
    vol_series = pd.Series(volume)
    vol_median = vol_series.rolling(window=20, min_periods=20).median().values
    volume_confirm = volume > (vol_median * 1.5)
    
    # Load weekly data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA50 for trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25
    
    # Start after warmup (need 20-period for Donchian)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_confirm[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = base_size
            else:
                signals[i] = -base_size
            continue
        
        # Long logic: break above Donchian high with volume confirmation and weekly uptrend
        long_condition = (close[i] > highest_high[i]) and volume_confirm[i] and (close[i] > ema_50_1w_aligned[i])
        # Short logic: break below Donchian low with volume confirmation and weekly downtrend
        short_condition = (close[i] < lowest_low[i]) and volume_confirm[i] and (close[i] < ema_50_1w_aligned[i])
        
        # Exit logic: opposite Donchian touch or trend reversal
        exit_long = (close[i] < lowest_low[i]) or (close[i] < ema_50_1w_aligned[i])
        exit_short = (close[i] > highest_high[i]) or (close[i] > ema_50_1w_aligned[i])
        
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

name = "1d_Donchian20_Breakout_WeeklyEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0