#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_Volume_Trend_1dTrend
Hypothesis: Price breaks 20-period Donchian channel on 12h with volume confirmation and 1d trend filter.
Works in bull markets (breakouts continue) and bear markets (breakdowns continue) by aligning with higher timeframe trend.
Target: 15-25 trades/year to minimize fee drift while capturing sustained moves.
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 12h Donchian channel (20 periods)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike: >1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)
    
    for i in range(start_idx, n):
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_max[i]
        lower = low_min[i]
        trend = ema_50_1d_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: break above upper channel with volume in uptrend
            if (price > upper and
                vol_spike and
                price > trend):
                signals[i] = 0.25
                position = 1
            # Short: break below lower channel with volume in downtrend
            elif (price < lower and
                  vol_spike and
                  price < trend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price re-enters channel or trend reverses
            if price < upper or price < trend:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price re-enters channel or trend reverses
            if price > lower or price > trend:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_Trend_1dTrend"
timeframe = "12h"
leverage = 1.0