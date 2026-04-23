#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d weekly pivot direction and volume confirmation.
- Long when price breaks above 6h Donchian upper (20) AND price > 1d weekly pivot R1 (bullish bias) AND volume > 1.5x 20-period average
- Short when price breaks below 6h Donchian lower (20) AND price < 1d weekly pivot S1 (bearish bias) AND volume > 1.5x 20-period average
- Exit when price reverts to 6h Donchian midpoint (mean reversion) OR weekly pivot bias flips
Uses 6h timeframe to target ~12-30 trades/year, minimizing fee drag while capturing structured breakouts.
Weekly pivot from 1d data provides multi-timeframe trend filter that adapts to bull/bear regimes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for weekly pivot - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's 1d data
    # Need to resample 1d to weekly manually since we only have daily data
    # Weekly high = max of last 7 daily highs, etc.
    weekly_high = pd.Series(high_1d).rolling(window=7, min_periods=7).max().values
    weekly_low = pd.Series(low_1d).rolling(window=7, min_periods=7).min().values
    weekly_close = pd.Series(close_1d).rolling(window=7, min_periods=7).last().values
    
    # Weekly pivot: PP = (WH+WL+WC)/3
    weekly_pp = (weekly_high + weekly_low + weekly_close) / 3
    weekly_range = weekly_high - weekly_low
    # Weekly R1 = PP + (WH -WL)*1.0/2, S1 = PP - (WH -WL)*1.0/2
    weekly_r1 = weekly_pp + weekly_range * 1.0 / 2
    weekly_s1 = weekly_pp - weekly_range * 1.0 / 2
    
    # Align weekly pivot to 6h timeframe
    weekly_pp_aligned = align_htf_to_ltf(prices, df_1d, weekly_pp)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    
    # 6h Donchian channels (20-period)
    donchian_window = 20
    donchian_upper = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    donchian_lower = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2
    
    # Volume average (20-period) on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(weekly_pp_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        pp_val = weekly_pp_aligned[i]
        r1_val = weekly_r1_aligned[i]
        s1_val = weekly_s1_aligned[i]
        upper_val = donchian_upper[i]
        lower_val = donchian_lower[i]
        mid_val = donchian_mid[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above Donchian upper AND price > weekly R1 (bullish bias) AND volume spike
            if (price > upper_val and price > r1_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower AND price < weekly S1 (bearish bias) AND volume spike
            elif (price < lower_val and price < s1_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price reverts to Donchian midpoint OR weekly bias turns bearish (price < S1)
                if price <= mid_val or price < s1_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price reverts to Donchian midpoint OR weekly bias turns bullish (price > R1)
                if price >= mid_val or price > r1_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_Donchian20_1dWeeklyPivot_Volume_Breakout"
timeframe = "6h"
leverage = 1.0