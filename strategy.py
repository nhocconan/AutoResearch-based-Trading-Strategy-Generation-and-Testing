#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + 1w EMA50 trend + volume spike
# Donchian channel breakout for trend capture with clear entry/exit levels
# 1w EMA50 for higher timeframe trend filter (only trade in trend direction)
# Volume spike (>1.5x 20-day average) for confirmation of breakout strength
# Exit when price crosses 10-day EMA (trend reversal signal)
# Designed for low-frequency, high-conviction trades to minimize fee drag
# Target: 15-25 trades/year (60-100 total over 4 years)
name = "1d_Donchian_1wEMA_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1d Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d Volume average (20-period)
    vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 1d EMA10 for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_20[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(ema_10[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-day average
        volume_filter = volume[i] > 1.5 * vol_20[i]
        
        if position == 0:
            # Long entry: price breaks above Donchian upper + above 1w EMA50 + volume spike
            if close[i] > high_20[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price breaks below Donchian lower + below 1w EMA50 + volume spike
            elif close[i] < low_20[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below 10-day EMA (trend weakening)
            if close[i] < ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above 10-day EMA (trend reversal)
            if close[i] > ema_10[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals