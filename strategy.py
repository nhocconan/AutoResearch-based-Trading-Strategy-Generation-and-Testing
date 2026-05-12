#!/usr/bin/env python3
name = "1d_Donchian20_Trend1w_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly trend filter: EMA34 on weekly close
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Weekly volume filter: volume > 1.5x 20-period average
    vol_ma_20_1w = pd.Series(df_1w['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_20_1w)
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for Donchian
    
    for i in range(start_idx, n):
        # Skip if weekly trend or volume data not ready
        if np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Entry conditions
        if position == 0:
            # Long: break above upper Donchian + weekly uptrend + volume confirmation
            if (high[i] > highest_high[i] and 
                close[i] > highest_high[i] and
                close[i] > ema34_1w_aligned[i] and  # weekly uptrend
                volume[i] > vol_ma_20_1w_aligned[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian + weekly downtrend + volume confirmation
            elif (low[i] < lowest_low[i] and 
                  close[i] < lowest_low[i] and
                  close[i] < ema34_1w_aligned[i] and  # weekly downtrend
                  volume[i] > vol_ma_20_1w_aligned[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below lower Donchian
            if low[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above upper Donchian
            if high[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals