#!/usr/bin/env python3
name = "4h_ThreeLevelBreakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

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
    
    # 1d trend filter: EMA34
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d volume filter: volume > 1.5x 20-period average
    vol_ma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # 4h price data for Three Level breakout
    high_4h = high
    low_4h = low
    close_4h = close
    
    # Calculate Three Level (similar to Camarilla but different multipliers)
    # Using 4h bars for the levels
    range_4h = high_4h - low_4h
    tlevel_h3 = close_4h + range_4h * 1.1 / 4  # H3 level
    tlevel_l3 = close_4h - range_4h * 1.1 / 4  # L3 level
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if daily trend or volume data not ready
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Tlevel H3 with daily uptrend and volume confirmation
            if (high[i] > tlevel_h3[i] and 
                close[i] > tlevel_h3[i] and
                close[i] > ema34_1d_aligned[i] and  # daily uptrend
                volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Tlevel L3 with daily downtrend and volume confirmation
            elif (low[i] < tlevel_l3[i] and 
                  close[i] < tlevel_l3[i] and
                  close[i] < ema34_1d_aligned[i] and  # daily downtrend
                  volume[i] > vol_ma_20_1d_aligned[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when price breaks below Tlevel L3 or reverses against trend
            if (low[i] < tlevel_l3[i] or 
                close[i] < ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when price breaks above Tlevel H3 or reverses against trend
            if (high[i] > tlevel_h3[i] or 
                close[i] > ema34_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals