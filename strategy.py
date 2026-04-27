#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Donchian(20) breakout with weekly EMA(50) trend filter and volume confirmation.
# Long when price breaks above 20-day high, above weekly EMA(50), and volume > 2x 10-day average.
# Short when price breaks below 20-day low, below weekly EMA(50), and volume > 2x 10-day average.
# Exit when price crosses opposite Donchian band or weekly EMA(50).
# Uses weekly trend to avoid whipsaws in bear markets while capturing long-term trends.
# Target: 7-25 trades/year (30-100 over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    close_weekly = df_weekly['close'].values
    
    # Calculate weekly EMA(50) for trend filter
    ema50_weekly = pd.Series(close_weekly).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema50_weekly)
    
    # Calculate daily Donchian(20) channels
    high_rolling_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_rolling_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: volume > 2x 10-day average
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    volume_filter = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_weekly_aligned[i]) or np.isnan(high_rolling_max[i]) or 
            np.isnan(low_rolling_min[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: break above 20-day high, above weekly EMA50, volume spike
        if (high[i] > high_rolling_max[i] and 
            close[i] > ema50_weekly_aligned[i] and 
            volume_filter[i]):
            signals[i] = 0.25
            position = 1
        # Short condition: break below 20-day low, below weekly EMA50, volume spike
        elif (low[i] < low_rolling_min[i] and 
              close[i] < ema50_weekly_aligned[i] and 
              volume_filter[i]):
            signals[i] = -0.25
            position = -1
        # Exit conditions: price crosses opposite Donchian band or weekly EMA50
        elif position == 1 and (low[i] < low_rolling_min[i] or close[i] < ema50_weekly_aligned[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (high[i] > high_rolling_max[i] or close[i] > ema50_weekly_aligned[i]):
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_Donchian20_1wEMA50_VolumeFilter"
timeframe = "1d"
leverage = 1.0