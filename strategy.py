#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h Donchian(20) breakout + 1w EMA(34) trend filter + volume confirmation.
    # Donchian breakout captures volatility expansion after consolidation.
    # 1w EMA(34) filter ensures we trade with the higher timeframe trend (more robust than lower TF).
    # Volume spike confirms breakout validity.
    # Discrete position sizing (0.0, ±0.25) minimizes fee churn.
    # Target: 75-200 total trades over 4 years (19-50/year).
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA trend filter (call ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Calculate 6h Donchian channels (20-period)
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 6h volume MA(20) for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1w EMA to 6h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.8 * 20-period MA
        volume_filter = volume[i] > 1.8 * volume_ma[i]
        
        # Trend filter: price above/below 1w EMA(34)
        uptrend = close[i] > ema_34_1w_aligned[i]
        downtrend = close[i] < ema_34_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = (close[i] > highest_20[i-1]) and volume_filter and uptrend
        short_breakout = (close[i] < lowest_20[i-1]) and volume_filter and downtrend
        
        # Exit conditions: price returns to midpoint of Donchian channel
        donchian_mid = (highest_20[i-1] + lowest_20[i-1]) / 2
        long_exit = close[i] < donchian_mid
        short_exit = close[i] > donchian_mid
        
        # Fixed position size (discrete levels to minimize fee churn)
        position_size = 0.25
        
        # Entry conditions
        if long_breakout and position != 1:
            position = 1
            signals[i] = position_size
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -position_size
        # Exit conditions
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        # Hold current position
        else:
            if position == 1:
                signals[i] = position_size
            elif position == -1:
                signals[i] = -position_size
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_donchian_breakout_ema_volume_v1"
timeframe = "6h"
leverage = 1.0