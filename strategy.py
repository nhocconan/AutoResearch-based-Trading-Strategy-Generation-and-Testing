#!/usr/bin/env python3
"""
1d_Donchian_Breakout_With_Volume_and_WeeklyTrend
Hypothesis: Buy when price breaks above weekly Donchian upper with volume surge and above 200-day EMA; short when breaks below weekly Donchian lower with volume surge and below 200-day EMA. Weekly trend filter ensures alignment with higher timeframe direction, while daily breakout captures momentum. Volume confirmation reduces false breakouts. Designed for low frequency (<15 trades/year) to minimize fee drag in both bull and bear markets.
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
    
    # Weekly Donchian channels (20-period high/low)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian channels: upper = max(high, 20), lower = min(low, 20)
    donch_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align to daily timeframe (wait for weekly close)
    donch_high_aligned = align_htf_to_ltf(prices, df_1w, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1w, donch_low)
    
    # Volume spike: >2.5x 50-period average (higher threshold for lower frequency)
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_spike = volume > (2.5 * vol_ma)
    
    # 200-day EMA trend filter (using daily close)
    ema_200 = pd.Series(close).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 200  # Need 200-day EMA and weekly Donchian
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or 
            np.isnan(donch_low_aligned[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(ema_200[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donch_high_aligned[i]
        lower = donch_low_aligned[i]
        vol_spike = volume_spike[i]
        ema_val = ema_200[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume surge and above 200-day EMA
            if price > upper and vol_spike and price > ema_val:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low with volume surge and below 200-day EMA
            elif price < lower and vol_spike and price < ema_val:
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            signals[i] = 0.30
            # Exit: price breaks below weekly Donchian low or below 200-day EMA
            if price < lower or price < ema_val:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.30
            # Exit: price breaks above weekly Donchian high or above 200-day EMA
            if price > upper or price > ema_val:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Donchian_Breakout_With_Volume_and_WeeklyTrend"
timeframe = "1d"
leverage = 1.0