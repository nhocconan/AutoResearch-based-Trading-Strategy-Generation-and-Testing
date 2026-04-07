#!/usr/bin/env python3
"""
1d_donchian_20_1w_trend_volume_v1
Hypothesis: On daily timeframe, use Donchian channel (20-day) breakout with weekly trend filter and volume confirmation. 
Enter long when price breaks above upper Donchian band with weekly trend up and volume > 1.5x average, short when breaks below lower band with weekly trend down and volume > 1.5x average.
Exit when price touches opposite Donchian band. Weekly trend uses 50-period EMA on weekly timeframe.
Designed for low frequency (7-25 trades/year) to minimize fee drift while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_20_1w_trend_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Calculate weekly EMA50 for trend
    weekly_close = df_1w['close'].values
    ema_50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Calculate daily Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper = high_series.rolling(window=20, min_periods=20).max().values
    lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-day average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if weekly data not available
        if np.isnan(ema_50_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Skip if Donchian bands not available
        if np.isnan(upper[i]) or np.isnan(lower[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = volume[i] > 1.5 * vol_avg[i] if not np.isnan(vol_avg[i]) else False
        
        # Weekly trend: price above/below EMA50
        weekly_up = close[i] > ema_50_aligned[i]
        weekly_down = close[i] < ema_50_aligned[i]
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian band
            if close[i] <= lower[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian band
            if close[i] >= upper[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian with weekly trend up and volume confirmation
            long_entry = (close[i] > upper[i]) and weekly_up and vol_confirm
            # Short entry: price breaks below lower Donchian with weekly trend down and volume confirmation
            short_entry = (close[i] < lower[i]) and weekly_down and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals