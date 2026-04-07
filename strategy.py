#!/usr/bin/env python3
"""
1d_donchian_breakout_1w_trend_v1
Hypothesis: Use Donchian channel breakout on daily timeframe with weekly EMA trend filter to capture strong directional moves. The Donchian breakout provides entry signals in the direction of the weekly trend, reducing whipsaws. ATR-based position sizing and stop-loss via signal=0 on reversal. Designed for low trade frequency (target: 30-100 trades over 4 years) to minimize fee drag while capturing major trends in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_donchian_breakout_1w_trend_v1"
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
    
    # Donchian channel parameters
    donchian_period = 20
    
    # Calculate Donchian channels
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Weekly EMA for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_period = 50
    ema_1w = pd.Series(close_1w).ewm(span=ema_period, adjust=False, min_periods=ema_period).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(donchian_period, n):
        # Skip if data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(close[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend from weekly EMA
        weekly_uptrend = close[i] > ema_1w_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price closes below Donchian lower band (trend reversal)
            if close[i] < lowest_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above Donchian upper band (trend reversal)
            if close[i] > highest_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only enter if weekly trend aligns
            if weekly_uptrend:
                # Long entry: price breaks above Donchian upper band
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = 0.25
            else:
                # Short entry: price breaks below Donchian lower band
                if close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals