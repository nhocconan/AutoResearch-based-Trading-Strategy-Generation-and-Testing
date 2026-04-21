#!/usr/bin/env python3
"""
1d_Donchian20_WeeklyTrend_Filter_v2
Hypothesis: On 1d timeframe, Donchian channel (20-period) breakouts aligned with weekly EMA34 trend regime capture strong directional moves with minimal whipsaw. 
Weekly trend filter (price > weekly EMA34 for longs, price < weekly EMA34 for shorts) ensures trades align with higher timeframe momentum. 
Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly trend)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # === 1w EMA34 for weekly trend regime ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 1d Donchian channel (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian upper/lower bands
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        weekly_ema = ema_34_1w_aligned[i]
        upper = donchian_upper[i]
        lower = donchian_lower[i]
        
        # Weekly trend regime
        is_bull = price > weekly_ema
        is_bear = price < weekly_ema
        
        if position == 0:
            # Bull regime: long when price breaks above Donchian upper band
            if is_bull and price > upper:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Bear regime: short when price breaks below Donchian lower band
            elif is_bear and price < lower:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position != 0:
            # Exit conditions: opposite Donchian band touch or weekly trend reversal
            if position == 1:  # long position
                # Exit if price touches lower band OR weekly trend turns bearish
                if price < lower or price < weekly_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1 (short position)
                # Exit if price touches upper band OR weekly trend turns bullish
                if price > upper or price > weekly_ema:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_WeeklyTrend_Filter_v2"
timeframe = "1d"
leverage = 1.0