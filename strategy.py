#!/usr/bin/env python3
"""
1h_Donchian20_Breakout_4hTrend_Volume
Hypothesis: On 1-hour timeframe, enter long when price breaks above 4h Donchian(20) high with volume surge and 4h uptrend (close > SMA50), short when price breaks below 4h Donchian(20) low with volume surge and 4h downtrend. Exit on opposite Donchian break. Uses 4h trend filter to avoid counter-trend trades. Designed for moderate trade frequency (~15-30/year) to minimize fee decay. Volume surge filters breakouts for institutional participation. Works in bull via trend-following and in bear via short signals during downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter and Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian(20) channels
    donch_high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # 4h SMA(50) for trend filter
    sma_50_4h = pd.Series(close_4h).rolling(window=50, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_4h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_4h, donch_low_20)
    sma_50_4h_aligned = align_htf_to_ltf(prices, df_4h, sma_50_4h)
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(sma_50_4h_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 4h trend alignment and volume surge
        long_entry = close[i] > donch_high_20_aligned[i] and close[i] > sma_50_4h_aligned[i] and volume_surge[i]
        short_entry = close[i] < donch_low_20_aligned[i] and close[i] < sma_50_4h_aligned[i] and volume_surge[i]
        
        # Exit on opposite Donchian break with volume surge
        long_exit = close[i] < donch_low_20_aligned[i] and volume_surge[i]
        short_exit = close[i] > donch_high_20_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.20
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.20
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.20  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.20   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
    
    return signals

name = "1h_Donchian20_Breakout_4hTrend_Volume"
timeframe = "1h"
leverage = 1.0