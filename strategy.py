#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d pivot direction filter and volume confirmation
# Uses daily pivot levels (from 1d data) to determine market bias: only long when price > daily pivot, short when price < daily pivot
# Donchian breakouts provide entry signals in the direction of daily bias
# Volume confirmation ensures breakouts have participation
# Works in both bull/bear: daily pivot adapts to intermediate-term trend, Donchian captures breakouts in ranging markets
# Target: 50-150 total trades over 4 years (12-37/year) to balance edge and fee drag

name = "12h_1d_donchian_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily pivot levels (standard floor pivot) using previous day's data
    daily_pivot = np.full(len(df_1d), np.nan)
    daily_r1 = np.full(len(df_1d), np.nan)
    daily_s1 = np.full(len(df_1d), np.nan)
    
    for i in range(len(df_1d)):
        if i < 1:
            daily_pivot[i] = np.nan
            daily_r1[i] = np.nan
            daily_s1[i] = np.nan
        else:
            # Use previous day's OHLC to calculate current day's pivot levels
            prev_high = df_1d['high'].iloc[i-1]
            prev_low = df_1d['low'].iloc[i-1]
            prev_close = df_1d['close'].iloc[i-1]
            
            pivot = (prev_high + prev_low + prev_close) / 3.0
            daily_pivot[i] = pivot
            daily_r1[i] = 2 * pivot - prev_low
            daily_s1[i] = 2 * pivot - prev_high
    
    # Align daily pivot levels to 12h timeframe
    daily_pivot_12h = align_htf_to_ltf(prices, df_1d, daily_pivot)
    daily_r1_12h = align_htf_to_ltf(prices, df_1d, daily_r1)
    daily_s1_12h = align_htf_to_ltf(prices, df_1d, daily_s1)
    
    # Calculate 20-period Donchian channels on 12h
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(n):
        if i < 20:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
        else:
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Calculate 20-period average volume for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(n):
        if i < 20:
            avg_volume[i] = np.nan
        else:
            avg_volume[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(daily_pivot_12h[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period average
        volume_confirm = volume[i] > 1.3 * avg_volume[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian low OR daily bias turns bearish (price < daily pivot)
            if close[i] < donchian_low[i] or close[i] < daily_pivot_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high OR daily bias turns bullish (price > daily pivot)
            if close[i] > donchian_high[i] or close[i] > daily_pivot_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic: Donchian breakout in direction of daily bias with volume confirmation
            if volume_confirm:
                # Long breakout: price closes above Donchian high AND price > daily pivot (bullish bias)
                if close[i] > donchian_high[i] and close[i] > daily_pivot_12h[i]:
                    position = 1
                    signals[i] = 0.25
                # Short breakout: price closes below Donchian low AND price < daily pivot (bearish bias)
                elif close[i] < donchian_low[i] and close[i] < daily_pivot_12h[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals