#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: On daily timeframe, use Donchian channel breakout with weekly trend filter and volume confirmation. 
Enter long when price breaks above 20-day high with weekly SMA > 50-period SMA and volume > 1.5x average, 
short when price breaks below 20-day low with weekly SMA < 50-period SMA and volume > 1.5x average. 
Exit when price touches opposite Donchian band. Designed for low frequency (7-25 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly SMAs for trend filter
    weekly_close = df_1w['close'].values
    sma_50 = pd.Series(weekly_close).rolling(window=50, min_periods=50).mean().values
    sma_200 = pd.Series(weekly_close).rolling(window=200, min_periods=200).mean().values
    weekly_trend_up = sma_50 > sma_200
    weekly_trend_down = sma_50 < sma_200
    
    # Align weekly trend to daily
    trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
    trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    
    # Calculate 20-day Donchian channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if weekly data not available
        if np.isnan(trend_up_aligned[i]) or np.isnan(trend_down_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Donchian channels not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        if position == 1:  # Long position
            # Exit when price touches or goes below Donchian low
            if close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above Donchian high
            if close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with weekly uptrend and volume confirmation
            long_entry = (close[i] > donchian_high[i]) and trend_up_aligned[i] and vol_confirm
            # Short entry: price breaks below Donchian low with weekly downtrend and volume confirmation
            short_entry = (close[i] < donchian_low[i]) and trend_down_aligned[i] and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals