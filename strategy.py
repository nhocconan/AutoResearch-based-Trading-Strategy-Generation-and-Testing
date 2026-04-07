#!/usr/bin/env python3
"""
12h_donchian_breakout_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use weekly Donchian channel breakouts for entry signals, filtered by weekly trend and volume confirmation.
- In uptrend (price > weekly EMA50): long when price breaks above upper Donchian(20) channel
- In downtrend (price < weekly EMA50): short when price breaks below lower Donchian(20) channel
Volume confirms genuine breakouts. Uses weekly timeframe for signal generation to reduce trade frequency and avoid overtrading.
Target: 12-37 trades/year (~50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_donchian_breakout_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Weekly data for Donchian channels and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema_50 = df_1w['close'].ewm(span=50, adjust=False).mean()
    
    # Align weekly EMA50 to 12h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50.values)
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = df_1w['high'].rolling(window=20, min_periods=20).max()
    low_20 = df_1w['low'].rolling(window=20, min_periods=20).min()
    
    # Align Donchian channels to 12h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1w, high_20.values)
    lower_20_aligned = align_htf_to_ltf(prices, df_1w, low_20.values)
    
    # Volume confirmation (24-period average on 12h = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(40, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price breaks below lower Donchian channel or trend turns bearish
            if close[i] < lower_20_aligned[i] or close[i] < ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price breaks above upper Donchian channel or trend turns bullish
            if close[i] > upper_20_aligned[i] or close[i] > ema_50_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian channel with volume in uptrend
            if (close[i] > upper_20_aligned[i] and
                vol_confirm and 
                close[i] > ema_50_aligned[i]):  # uptrend filter
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below lower Donchian channel with volume in downtrend
            elif (close[i] < lower_20_aligned[i] and
                  vol_confirm and 
                  close[i] < ema_50_aligned[i]):  # downtrend filter
                position = -1
                signals[i] = -0.25
    
    return signals