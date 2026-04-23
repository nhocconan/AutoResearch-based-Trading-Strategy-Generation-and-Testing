#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA34 trend filter and volume spike confirmation.
Long when price breaks above Donchian upper channel AND 12h EMA34 uptrend AND volume > 2.0x 20-period MA.
Short when price breaks below Donchian lower channel AND 12h EMA34 downtrend AND volume > 2.0x 20-period MA.
Exit when price crosses Donchian middle (20-period SMA) or opposite Donchian breakout.
Designed for ~25-35 trades/year with strong edge via institutional breakout follow-through.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Donchian channels (20-period) on 4h
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_middle = (highest_high_20 + lowest_low_20) / 2.0
    
    # Calculate volume MA (20-period) for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 20)  # need EMA34, Donchian20, volume MA20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(highest_high_20[i]) or 
            np.isnan(lowest_low_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: 12h close > EMA34 = uptrend, close < EMA34 = downtrend
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, close_12h)
        trend_up = close_12h_aligned[i] > ema_34_12h_aligned[i]
        trend_down = close_12h_aligned[i] < ema_34_12h_aligned[i]
        
        # Volume filter: 4h volume > 2.0x 20-period MA (strong confirmation)
        vol_filter = volume[i] > 2.0 * vol_ma_20[i]
        
        # Donchian breakout conditions
        breakout_up = close[i] > highest_high_20[i-1]  # break above upper channel
        breakout_down = close[i] < lowest_low_20[i-1]  # break below lower channel
        middle_cross = (position == 1 and close[i] < donchian_middle[i]) or \
                       (position == -1 and close[i] > donchian_middle[i])
        opposite_breakout = (position == 1 and breakout_down) or \
                            (position == -1 and breakout_up)
        
        if position == 0:
            # Long: Donchian breakout up AND uptrend AND volume spike
            if breakout_up and trend_up and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND downtrend AND volume spike
            elif breakout_down and trend_down and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: middle cross or opposite breakout
            exit_signal = middle_cross or opposite_breakout
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_12hEMA34_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0