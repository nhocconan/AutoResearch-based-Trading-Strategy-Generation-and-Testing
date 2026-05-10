#!/usr/bin/env python3
# 12h_Donchian_Breakout_20_200MA_Volume_Confirmation
# Hypothesis: Use 12h Donchian(20) breakouts with 200-period MA trend filter and volume confirmation.
# Works in bull markets via breakouts above Donchian high, in bear markets via breakdowns below Donchian low.
# Low-frequency design targets 20-40 trades per year to minimize fee drag. Uses 1d HTF for 200MA.

name = "12h_Donchian_Breakout_20_200MA_Volume_Confirmation"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) for breakout signals
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # 200-period MA from 1d timeframe (trend filter)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ma_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ma_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ma_200_1d)
    
    # Volume confirmation: 20-period average volume
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Need enough history for 200MA
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ma_200_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: price above/below 200MA
        is_uptrend = close[i] > ma_200_1d_aligned[i]
        is_downtrend = close[i] < ma_200_1d_aligned[i]
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian high in uptrend with volume
            if is_uptrend and close[i] > donchian_high[i] and volume_condition:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low in downtrend with volume
            elif is_downtrend and close[i] < donchian_low[i] and volume_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian low or trend turns down
            if close[i] < donchian_low[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian high or trend turns up
            if close[i] > donchian_high[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals