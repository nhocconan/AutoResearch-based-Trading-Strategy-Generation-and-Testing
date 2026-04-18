#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeTrend_4HR
Hypothesis: Donchian(20) breakout on 4h with volume confirmation and 1d EMA trend filter.
Breakouts above upper band signal long, below lower band signal short when confirmed by volume spike (>2x 20-period average) and aligned with daily trend.
Designed for low trade frequency (20-50/year) to avoid fee drag while capturing strong trending moves in both bull and bear markets.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels on 4h (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(35, 20)  # Warmup for EMA and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema34 = ema_34_aligned[i]
        vol_spike = volume_spike[i]
        upper = high_20[i]
        lower = low_20[i]
        
        if position == 0:
            # Long: price breaks above upper Donchian band with volume spike and uptrend
            if price > upper and vol_spike and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian band with volume spike and downtrend
            elif price < lower and vol_spike and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian band OR trend turns down
            if price < lower:
                signals[i] = 0.0
                position = 0
            elif price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian band OR trend turns up
            if price > upper:
                signals[i] = 0.0
                position = 0
            elif price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian20_Breakout_VolumeTrend_4HR"
timeframe = "4h"
leverage = 1.0