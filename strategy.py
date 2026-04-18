#!/usr/bin/env python3
"""
1d_WeeklyTrend_Follower
Hypothesis: On the daily chart, capture trending moves in both bull and bear markets by combining:
- Weekly EMA50 as the primary trend filter (only trade long when price > weekly EMA50, short when price < weekly EMA50)
- Daily Donchian(20) breakouts for entry timing
- Volume confirmation (>1.5x 20-day average volume) to filter false breakouts
This approach aims for fewer, higher-quality trades by requiring alignment across timeframes and volume confirmation,
making it suitable for trending markets while avoiding choppy periods. Target: 15-25 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly EMA50 for trend filter (loaded once before loop)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily Donchian channels (20-period)
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: >1.5x 20-day average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # Warmup for weekly EMA50 and Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or 
            np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema50 = ema_50_1w_aligned[i]
        upper = high_max[i]
        lower = low_min[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper with volume confirmation and uptrend (price > weekly EMA50)
            if price > upper and vol_conf and price > ema50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower with volume confirmation and downtrend (price < weekly EMA50)
            elif price < lower and vol_conf and price < ema50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price closes below Donchian lower (reversal) or weekly EMA50 (trend change)
            if price < lower or price < ema50:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price closes above Donchian upper (reversal) or weekly EMA50 (trend change)
            if price > upper or price > ema50:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_WeeklyTrend_Follower"
timeframe = "1d"
leverage = 1.0