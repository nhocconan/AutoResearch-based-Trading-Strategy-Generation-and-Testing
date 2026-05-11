#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrend_VolumeSMA
Hypothesis: Donchian(20) breakout with weekly trend filter (EMA34) and volume confirmation (2x SMA20).
Long when price breaks above 20-bar high + weekly uptrend + volume spike.
Short when price breaks below 20-bar low + weekly downtrend + volume spike.
Designed for low trade frequency (<15/year) on 12h timeframe to minimize fee drag.
Works in bull/bear by using weekly trend to filter direction, volume for confirmation.
"""

name = "12h_Donchian20_Breakout_1wTrend_VolumeSMA"
timeframe = "12h"
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
    
    # === Weekly EMA34 Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_12h = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Donchian Channel (20-period) ===
    # Calculate rolling high/low with min_periods=20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # === Volume Filter (2x SMA20) ===
    volume_series = pd.Series(volume)
    volume_sma20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > volume_sma20 * 2.0
    
    # === Signal Parameters ===
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (covers Donchian and EMA calculation)
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema34_1w_12h[i]) or np.isnan(volume_ok[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close breaks above Donchian high + weekly uptrend + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema34_1w_12h[i] and volume_ok[i]):
                signals[i] = position_size
                position = 1
            # Short: Close breaks below Donchian low + weekly downtrend + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema34_1w_12h[i] and volume_ok[i]):
                signals[i] = -position_size
                position = -1
        else:
            # Exit: Close crosses back through the opposite Donchian level
            if position == 1:
                if close[i] < donchian_low[i]:  # Exit long if price breaks below Donchian low
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if close[i] > donchian_high[i]:  # Exit short if price breaks above Donchian high
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals