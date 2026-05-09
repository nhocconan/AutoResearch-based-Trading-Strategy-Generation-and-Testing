#!/usr/bin/env python3
# 1h_4H_1D_Combo_Signal
# Uses 4h Donchian channel breakout for signal direction, 1d volume spike for confirmation, and 1h for precise entry timing.
# Designed to work in both bull and bear markets: Donchian breakouts capture trends, volume filters avoid false breakouts in ranging markets.
# Target: 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.

name = "1h_4H_1D_Combo_Signal"
timeframe = "1h"
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
    
    # 4h Donchian channel (20-period) for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h Donchian upper and lower bands
    donch_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align to 1h timeframe (waits for 4h bar close)
    donch_high_aligned = align_htf_to_ltf(prices, df_4h, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_4h, donch_low)
    
    # 1d volume confirmation: current volume > 2.0x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_ma_1d = pd.Series(df_1d['volume']).rolling(window=20, min_periods=20).mean().values
    vol_spike_1d = df_1d['volume'].values > (2.0 * vol_ma_1d)
    
    # Align 1d volume spike to 1h (waits for daily close)
    vol_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_spike_1d.astype(float))
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready or outside session
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(vol_spike_1d_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above 4h Donchian high + volume spike
            if close[i] > donch_high_aligned[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = 0.20
                position = 1
            # Enter short: price breaks below 4h Donchian low + volume spike
            elif close[i] < donch_low_aligned[i] and vol_spike_1d_aligned[i] > 0.5:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h Donchian low
            if close[i] < donch_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h Donchian high
            if close[i] > donch_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals