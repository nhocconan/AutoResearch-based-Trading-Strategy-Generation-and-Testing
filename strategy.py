#!/usr/bin/env python3
# 1d_weekly_donchian_breakout_v1
# Hypothesis: Uses weekly Donchian channel breakout with weekly trend filter and volume confirmation.
# Enters long when price breaks above weekly Donchian high with volume spike and weekly uptrend.
# Enters short when price breaks below weekly Donchian low with volume spike and weekly downtrend.
# Exits on opposite break or trend failure. Designed for 15-25 trades/year to avoid fee drag.
# Uses weekly trend filter for multi-timeframe alignment and weekly Donchian channels as structure.
# Should work in both bull and bear markets via weekly trend alignment and volume confirmation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for Donchian channels and trend
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    # Upper band = highest high of last 20 weeks
    # Lower band = lowest low of last 20 weeks
    donchian_high = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align weekly data to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, donchian_low)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily volume average (20-period)
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or np.isnan(ema50_1w_aligned[i]) or np.isnan(avg_volume[i]):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter
        weekly_uptrend = close[i] > ema50_1w_aligned[i]
        weekly_downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 2.0 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: breakdown below weekly Donchian low or weekly trend failure
            if close[i] < donchian_low_aligned[i] or not weekly_uptrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: break above weekly Donchian high or weekly trend failure
            if close[i] > donchian_high_aligned[i] or not weekly_downtrend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long entry: above weekly Donchian high with volume spike and weekly uptrend
                if close[i] > donchian_high_aligned[i] and weekly_uptrend:
                    position = 1
                    signals[i] = 0.25
                # Short entry: below weekly Donchian low with volume spike and weekly downtrend
                elif close[i] < donchian_low_aligned[i] and weekly_downtrend:
                    position = -1
                    signals[i] = -0.25
    
    return signals