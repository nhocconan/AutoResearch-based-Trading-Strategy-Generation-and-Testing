#!/usr/bin/env python3
"""
6h_Donchian_Breakout_Volume_1dTrend
Hypothesis: Buy when price breaks above 6h Donchian(20) high with volume spike and above 1d EMA50; short when breaks below 6h Donchian(20) low with volume spike and below 1d EMA50. Donchian breakouts capture momentum, volume confirms institutional participation, and 1d EMA50 ensures alignment with daily trend. Designed for low trade frequency to minimize fee drag while capturing high-probability breakouts.
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
    
    # 6h Donchian(20) - need 20 periods of high/low
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 1d EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume spike: >2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 40  # Need Donchian, volume MA, and EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or
            np.isnan(ema_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema_trend = ema_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price > upper Donchian with volume spike and above 1d EMA50
            if price > upper and vol_spike and price > ema_trend:
                signals[i] = 0.25
                position = 1
            # Short: price < lower Donchian with volume spike and below 1d EMA50
            elif price < lower and vol_spike and price < ema_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price < lower Donchian or below 1d EMA50
            if price < lower or price < ema_trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price > upper Donchian or above 1d EMA50
            if price > upper or price > ema_trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_Donchian_Breakout_Volume_1dTrend"
timeframe = "6h"
leverage = 1.0