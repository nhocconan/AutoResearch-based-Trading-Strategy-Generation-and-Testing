#!/usr/bin/env python3
"""
4h_donchian_20_1d_trend_volume_v6
Hypothesis: On 4-hour timeframe, use Donchian channel (20) breakout with 1-day trend filter and volume confirmation.
Enter long when price breaks above Donchian upper band with price above 1-day EMA(50) and volume > 1.5x average.
Enter short when price breaks below Donchian lower band with price below 1-day EMA(50) and volume > 1.5x average.
Exit when price touches opposite Donchian band or trend reverses.
Designed for low frequency (20-50 trades/year) to minimize fee drag while capturing strong breakouts with trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_donchian_20_1d_trend_volume_v6"
timeframe = "4h"
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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if daily EMA not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or np.isnan(vol_avg[i]):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        if position == 1:  # Long position
            # Exit when price touches or goes below lower Donchian band OR trend turns bearish
            if close[i] <= donchian_lower[i] or close[i] < ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit when price touches or goes above upper Donchian band OR trend turns bullish
            if close[i] >= donchian_upper[i] or close[i] > ema_50_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above upper Donchian band with price above 1-day EMA(50) and volume confirmation
            long_entry = (close[i] > donchian_upper[i]) and (close[i] > ema_50_1d_aligned[i]) and vol_confirm
            # Short entry: price breaks below lower Donchian band with price below 1-day EMA(50) and volume confirmation
            short_entry = (close[i] < donchian_lower[i]) and (close[i] < ema_50_1d_aligned[i]) and vol_confirm
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals