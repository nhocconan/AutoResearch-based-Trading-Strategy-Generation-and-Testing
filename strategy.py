#!/usr/bin/env python3
name = "6h_Donchian20_1dTrend_VolumeSpike"
timeframe = "6h"
leverage = 1.0

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
    
    # Donchian channel (20-period) - breakout levels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Get daily data for trend filter and volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily EMA34 trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily volume spike: current 6h volume > 2.0 x average daily volume per 6h bar
    # Approximate: daily volume / 4 (since 6h bars per day = 4)
    vol_1d = df_1d['volume'].values
    vol_ma_20d = pd.Series(vol_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    volume_threshold = vol_ma_20d_aligned / 4.0 * 2.0  # 2x average 6h volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2x average 6h volume
        vol_condition = volume[i] > volume_threshold[i]
        
        if position == 0:
            # LONG: Break above Donchian high with daily uptrend and volume spike
            if close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below Donchian low with daily downtrend and volume spike
            elif close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price re-enters Donchian channel (below Donchian high) or trend reversal
            if close[i] < donchian_high[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price re-enters Donchian channel (above Donchian low) or trend reversal
            if close[i] > donchian_low[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals