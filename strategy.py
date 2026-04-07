#!/usr/bin/env python3
"""
6h_donchian_20_12h_trend_volume_v1
Hypothesis: On 6-hour timeframe, combine Donchian channel breakout with 12-hour trend filter and volume confirmation.
Enter long when price breaks above 20-period high with 12h EMA uptrend and volume > 1.5x average.
Enter short when price breaks below 20-period low with 12h EMA downtrend and volume > 1.5x average.
Exit when price touches opposite Donchian level or trend reverses.
Designed for low frequency (12-37 trades/year) to minimize fee drag while capturing strong breakouts.
Works in both bull and bear markets via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_20_12h_trend_volume_v1"
timeframe = "6h"
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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian and volume average warmup
        # Skip if data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_12h_aligned[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit when price touches or goes below Donchian low OR trend turns down
            if close[i] <= donchian_low[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above Donchian high OR trend turns up
            if close[i] >= donchian_high[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above Donchian high with uptrend and volume confirmation
            long_entry = (close[i] > donchian_high[i]) and (close[i] > ema_12h_aligned[i]) and vol_confirm
            # Short entry: price breaks below Donchian low with downtrend and volume confirmation
            short_entry = (close[i] < donchian_low[i]) and (close[i] < ema_12h_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals