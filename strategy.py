#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v1
Hypothesis: On 4-hour timeframe, use Donchian channel breakout with daily trend filter and volume confirmation.
Enter long when price breaks above 20-period high with daily EMA50 uptrend and volume > 1.5x average.
Enter short when price breaks below 20-period low with daily EMA50 downtrend and volume > 1.5x average.
Exit when price touches opposite Donchian band or trend reverses.
Designed for low frequency (20-50 trades/year) to minimize fee drag while capturing strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 20-period Donchian channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian band OR trend turns down
            if close[i] <= low_roll[i] or close[i] < ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian band OR trend turns up
            if close[i] >= high_roll[i] or close[i] > ema50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with daily uptrend and volume confirmation
            long_entry = (close[i] > high_roll[i]) and (close[i] > ema50_1d_aligned[i]) and vol_confirm
            # Short entry: price breaks below lower Donchian band with daily downtrend and volume confirmation
            short_entry = (close[i] < low_roll[i]) and (close[i] < ema50_1d_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals