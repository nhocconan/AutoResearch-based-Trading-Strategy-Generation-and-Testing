#!/usr/bin/env python3
"""
6H Donchian Breakout + Weekly Trend + Volume Confirmation
Hypothesis: Donchian channel breakouts from weekly trend direction with volume confirmation capture strong momentum.
Weekly trend filter prevents counter-trend trades, improving performance in both bull and bear markets.
Target: 20-40 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_trend_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    weekly_ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False).mean().values
    weekly_ema_50_6h = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    # Daily data for Donchian channel (20-day high/low)
    df_1d = get_htf_data(prices, '1d')
    donchian_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    donchian_high_6h = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_6h = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Volume filter (>1.5x 20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_6h[i]) or np.isnan(donchian_low_6h[i]) or 
            np.isnan(weekly_ema_50_6h[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA50 or Donchian lower band
            if close[i] < weekly_ema_50_6h[i] or close[i] < donchian_low_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA50 or Donchian upper band
            if close[i] > weekly_ema_50_6h[i] or close[i] > donchian_high_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Breakout long above Donchian high with weekly uptrend and volume
            if (close[i] >= donchian_high_6h[i] and 
                close[i] > weekly_ema_50_6h[i] and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Breakout short below Donchian low with weekly downtrend and volume
            elif (close[i] <= donchian_low_6h[i] and 
                  close[i] < weekly_ema_50_6h[i] and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals