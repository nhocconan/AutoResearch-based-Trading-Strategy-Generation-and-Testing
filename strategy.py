#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND 1w EMA50 rising AND volume > 1.5x 20-period MA.
Short when price breaks below Donchian(20) low AND 1w EMA50 falling AND volume > 1.5x 20-period MA.
Exit when price touches the opposite Donchian(20) level or EMA50 reverses.
Uses 1w HTF for trend filter to avoid counter-trend trades, volume spike for momentum confirmation.
Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 12h Donchian(20) channels
    donchian_period = 20
    donchian_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Calculate 12h volume MA (20-period) for spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, donchian_period, 20)  # EMA50, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate EMA50 slope for trend direction (rising/falling)
        if i >= start_idx + 1:
            ema_prev = ema_50_1w_aligned[i-1]
            ema_rising = ema_50_1w_aligned[i] > ema_prev
            ema_falling = ema_50_1w_aligned[i] < ema_prev
        else:
            ema_rising = False
            ema_falling = False
        
        # Volume filter: 12h volume > 1.5x 20-period MA (strict threshold to reduce trades)
        vol_filter = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND EMA50 rising AND volume filter
            if close[i] > donchian_high[i] and ema_rising and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND EMA50 falling AND volume filter
            elif close[i] < donchian_low[i] and ema_falling and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Long exit: price touches Donchian low OR EMA50 starts falling
                if close[i] <= donchian_low[i] or (i >= start_idx + 1 and ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):
                    exit_signal = True
            elif position == -1:
                # Short exit: price touches Donchian high OR EMA50 starts rising
                if close[i] >= donchian_high[i] or (i >= start_idx + 1 and ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1wEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0