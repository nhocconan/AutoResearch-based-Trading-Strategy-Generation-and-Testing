#!/usr/bin/env python3
"""
12h_Donchian_20_Breakout_With_Volume_And_1dTrend
Hypothesis: Breakout trading on 12h timeframe using Donchian channels (20-period) with volume confirmation and 1d trend filter.
In bull markets, price breaks above upper Donchian channel with volume surge; in bear markets, breaks below lower channel with volume.
The 1d EMA50 trend filter ensures we only take trades aligned with higher timeframe trend, reducing false breakouts.
Low trade frequency expected (~20-40/year) to minimize fee drag while capturing strong momentum moves.
Works in both bull (breakouts continuation) and bear (breakdown continuation) markets.
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
    
    # Donchian Channel (20-period)
    donch_len = 20
    highest_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lowest_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Volume confirmation: >1.8x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_spike = volume > (1.8 * vol_ma)
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(50, donch_len, 30)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_donch = highest_high[i]
        lower_donch = lowest_low[i]
        vol_spike = volume_spike[i]
        ema_1d_val = ema_1d_aligned[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume spike and above 1d EMA50
            if price > upper_donch and vol_spike and price > ema_1d_val:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume spike and below 1d EMA50
            elif price < lower_donch and vol_spike and price < ema_1d_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price re-enters Donchian channel (middle) or trend reversal
            mid_point = (upper_donch + lower_donch) * 0.5
            if price < mid_point or price < ema_1d_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price re-enters Donchian channel or trend reversal
            mid_point = (upper_donch + lower_donch) * 0.5
            if price > mid_point or price > ema_1d_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian_20_Breakout_With_Volume_And_1dTrend"
timeframe = "12h"
leverage = 1.0