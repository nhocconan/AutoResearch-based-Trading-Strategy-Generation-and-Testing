#!/usr/bin/env python3
"""
6h_Donchian20_WeeklyTrend_Filter_v1
Hypothesis: On 6h timeframe, Donchian(20) breakouts aligned with weekly trend (price > weekly EMA50 for longs, < weekly EMA50 for shorts) capture institutional momentum. Weekly trend filter reduces false breakouts in ranging markets. Uses discrete sizing (0.25) to minimize fee churn. Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (weekly for trend filter)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA50 for weekly trend filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # === 6h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower (20-period high/low)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        weekly_ema = ema_50_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above Donchian high + above weekly EMA50
            if price > upper and price > weekly_ema:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + below weekly EMA50
            elif price < lower and price < weekly_ema:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price re-enters Donchian channel or weekly trend reverses
            if position == 1:
                if price < upper or price < weekly_ema:  # Re-enter channel or trend reversal
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > lower or price > weekly_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_WeeklyTrend_Filter_v1"
timeframe = "6h"
leverage = 1.0