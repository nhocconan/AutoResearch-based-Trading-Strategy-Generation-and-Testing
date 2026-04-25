#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike
Hypothesis: Donchian(20) breakout on 12h with volume spike and 1w EMA34 trend filter captures strong momentum moves. Uses 1w EMA for stronger trend bias suitable for 12h timeframe, reducing whipsaws in ranging markets. Discrete position sizing (0.25) to control fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1w data for EMA trend filter (loaded ONCE)
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Donchian(20) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need Donchian(20) + EMA (34) + volume MA (20)
    start_idx = max(20, 34, 20) + 5  # Conservative warmup
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        if position == 0:
            # Long: price breaks above Donchian high with volume spike and 1w uptrend
            long_breakout = (curr_close > highest_high[i]) and vol_spike[i] and (curr_close > ema_aligned[i])
            # Short: price breaks below Donchian low with volume spike and 1w downtrend
            short_breakout = (curr_close < lowest_low[i]) and vol_spike[i] and (curr_close < ema_aligned[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend turns down
            if (curr_close < lowest_low[i]) or (curr_close < ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend turns up
            if (curr_close > highest_high[i]) or (curr_close > ema_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1wEMA34_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0