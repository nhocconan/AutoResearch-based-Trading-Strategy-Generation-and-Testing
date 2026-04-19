#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# - Long when price breaks above 20-day high + price > weekly EMA50 + volume surge
# - Short when price breaks below 20-day low + price < weekly EMA50 + volume surge
# - Exit on opposite breakout or trend reversal
# - Designed to capture trends in both bull and bear markets by following higher timeframe trend
# - Target: 15-25 trades/year to minimize fee drag

name = "1d_Donchian20_1wTrend_Volume_v1"
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
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # 1w EMA(50) for trend direction
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 1.5x 20-day average
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        if np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Look for long entry: uptrend + breakout above 20-day high + volume
            if close[i] > ema_50_1w_aligned[i] and close[i] > highest_high[i-1] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: downtrend + breakdown below 20-day low + volume
            elif close[i] < ema_50_1w_aligned[i] and close[i] < lowest_low[i-1] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown below 20-day low or trend reversal
            if close[i] < lowest_low[i-1] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout above 20-day high or trend reversal
            if close[i] > highest_high[i-1] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals