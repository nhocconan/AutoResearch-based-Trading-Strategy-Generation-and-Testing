#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1w EMA200 trend filter and volume confirmation
# Uses 1w HTF for EMA200 to capture long-term trend and avoid counter-trend trades.
# Donchian(20) from prior completed 6h bar provides breakout levels.
# Volume confirmation at 2.0x average ensures strong participation.
# Session filter (08-20 UTC) reduces noise during low liquidity.
# Discrete sizing 0.25 to minimize fee churn. Works in bull/bear: long-term trend filter ensures trades only with major momentum.
# Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity and fee drag.

name = "6h_Donchian20_Breakout_1wEMA200_Volume"
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
    open_time = prices['open_time'].values
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate Donchian levels from prior completed 6h bar (shift by 1)
    if len(prices) < 2:
        return np.zeros(n)
    
    # Get prior completed 6h bar's high/low (shift by 1 for 6h timeframe)
    prev_high_6h = prices['high'].shift(1).values
    prev_low_6h = prices['low'].shift(1).values
    
    # Donchian(20) upper/lower bands from prior completed bars
    high_ma = pd.Series(prev_high_6h).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(prev_low_6h).rolling(window=20, min_periods=20).min().values
    
    # 1w EMA200 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Volume confirmation: 2.0x 20-period average (strict threshold to reduce trades)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for all indicators)
    start_idx = 200  # Need enough for 1w EMA200
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or 
            np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long: Price breaks above Donchian upper AND price > 1w EMA200 AND volume spike
            if (close[i] > high_ma[i] and 
                close[i] > ema_200_1w_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower AND price < 1w EMA200 AND volume spike
            elif (close[i] < low_ma[i] and 
                  close[i] < ema_200_1w_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price drops below Donchian lower OR price < 1w EMA200
            if close[i] < low_ma[i] or close[i] < ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit: Price rises above Donchian upper OR price > 1w EMA200
            if close[i] > high_ma[i] or close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals