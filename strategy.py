#!/usr/bin/env python3
"""
4h_Donchian_Breakout_VolumeTrend
Hypothesis: Breakouts from Donchian(20) channels with volume confirmation and trend filtering (EMA34) work on 4h timeframe.
Trades only in direction of 1d EMA34 trend to avoid counter-trend trades. Uses volume > 1.5x 20-period average for confirmation.
Designed for moderate trade frequency (20-50/year) to balance opportunity and fee drag while capturing directional moves.
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
    close_1d = df_1d['close'].values
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(34) for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 34)  # Warmup for Donchian and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_aligned[i]) or np.isnan(volume_confirm[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        ema34 = ema_34_aligned[i]
        vol_confirm = volume_confirm[i]
        
        if position == 0:
            # Long: break above upper Donchian with volume confirmation and uptrend
            if price > upper and vol_confirm and price > ema34:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian with volume confirmation and downtrend
            elif price < lower and vol_confirm and price < ema34:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price breaks below lower Donchian OR trend turns down
            if price < lower or price < ema34:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price breaks above upper Donchian OR trend turns up
            if price > upper or price > ema34:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_VolumeTrend"
timeframe = "4h"
leverage = 1.0