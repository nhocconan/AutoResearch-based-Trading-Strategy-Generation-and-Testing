#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian(20) breakout with weekly trend filter and volume confirmation
# - Long when price breaks above 20-day high + price above weekly EMA50 + volume > 1.5x 20-day average
# - Short when price breaks below 20-day low + price below weekly EMA50 + volume > 1.5x 20-day average
# - Exit when price crosses back below/above 10-day moving average or trend reverses
# - Weekly trend filter ensures alignment with higher timeframe direction
# - Volume confirmation adds conviction to breakouts
# - Designed for low-frequency, high-conviction trades to minimize fee drag
# - Target: 15-25 trades/year to stay within optimal range for 1d timeframe

name = "1d_Donchian20_WeeklyTrend_Volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily indicators
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-day EMA for exit signal
    ema_10 = pd.Series(close).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # 20-day average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_10[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average volume
        volume_filter = vol_ma_20[i] > 0 and volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Look for long entry: price breaks above 20-day high + uptrend + volume
            if close[i] > highest_high[i] and close[i] > ema_50_1w_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below 20-day low + downtrend + volume
            elif close[i] < lowest_low[i] and close[i] < ema_50_1w_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit when price crosses below 10-day EMA or trend reverses
            if close[i] < ema_10[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit when price crosses above 10-day EMA or trend reverses
            if close[i] > ema_10[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals