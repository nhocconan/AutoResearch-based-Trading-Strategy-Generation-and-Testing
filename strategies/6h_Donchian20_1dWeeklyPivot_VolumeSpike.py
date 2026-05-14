#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
# Targets 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Donchian channels provide clear breakout levels with proven structure
# 1d weekly pivot determines longer-term bias: long when price > weekly pivot, short when price < weekly pivot
# Volume spike (1.5x 20-period average) confirms institutional participation
# Works in bull markets via breakouts with trend alignment and bear markets via fade of false breakouts
# Discrete position sizing: 0.25 (25% of capital) balances exposure and risk

name = "6h_Donchian20_1dWeeklyPivot_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d data for weekly pivot and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points from prior completed 1d bar's weekly data
    # Using prior 5 completed 1d bars to approximate weekly OHLC
    if len(df_1d) >= 5:
        # Get last 5 completed 1d bars (prior week)
        week_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1).values
        week_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1).values
        week_close = pd.Series(df_1d['close']).shift(1).values
        
        # Weekly pivot: (H + L + C) / 3
        weekly_pivot = (week_high + week_low + week_close) / 3.0
        
        # Weekly R1 and S1: 2*P - L and 2*P - H
        weekly_r1 = 2 * weekly_pivot - week_low
        weekly_s1 = 2 * weekly_pivot - week_high
        
        # Align to 6h timeframe (wait for completed 1d bar)
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
    else:
        return np.zeros(n)
    
    # Calculate 6h Donchian(20) channels (prior completed 6h bar's range)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Calculate 6h volume spike (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().shift(1).values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Check for NaN values in indicators
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: price breaks above Donchian high AND price > weekly pivot (bullish bias) AND volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > weekly_pivot_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian low AND price < weekly pivot (bearish bias) AND volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < weekly_pivot_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: price falls below Donchian low OR below weekly pivot (trend change)
            if close[i] < donchian_low[i] or close[i] < weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: price rises above Donchian high OR above weekly pivot (trend change)
            if close[i] > donchian_high[i] or close[i] > weekly_pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals