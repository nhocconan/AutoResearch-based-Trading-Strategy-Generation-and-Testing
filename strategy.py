# %%
#!/usr/bin/env python3
"""
1d_Donchian20_Breakout_1wTrend_Volume
Hypothesis: Daily Donchian(20) breakout in direction of weekly EMA50 trend, with volume confirmation.
Donchian channels capture breakouts from consolidation, weekly trend filters ensure alignment with higher timeframe momentum.
Volume confirmation reduces false breakouts. Works in both bull and bear markets by following weekly trend.
Target: 50-100 total trades over 4 years (12-25/year) to minimize fee drag.
"""

name = "1d_Donchian20_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50 = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Daily price and volume
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian Channel (20-period)
    # Upper band: highest high of last 20 days
    # Lower band: lowest low of last 20 days
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.5x 20-day EMA of volume
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need Donchian (20), weekly EMA (50), volume EMA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above weekly EMA50 (uptrend) AND price breaks above Donchian Upper with volume
            if close[i] > ema_50_aligned[i] and high[i] > donchian_upper[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below weekly EMA50 (downtrend) AND price breaks below Donchian Lower with volume
            elif close[i] < ema_50_aligned[i] and low[i] < donchian_lower[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian Lower OR trend turns bearish
            if low[i] < donchian_lower[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian Upper OR trend turns bullish
            if high[i] > donchian_upper[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals
# %%