#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d trend filter + volume confirmation + ATR stoploss
# - 4h Donchian breakout: long when price breaks above 20-period high, short when breaks below 20-period low
# - 1d EMA(50) trend filter: only take long when close > 1d EMA50, short when close < 1d EMA50
# - Volume filter: current 4h volume > 1.5x 20-period average of 4h volume
# - Exit: opposite Donchian breakout or trend reversal
# - Position size: 0.25 (25%) to manage drawdown
# - Target: 20-50 trades/year to avoid excessive fee drag

name = "4h_Donchian20_1dTrend_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume average (20-period)
    vol_ma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for Donchian channels
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_4h[i]):
            signals[i] = 0.0
            continue
            
        # Volume filter: current 4h volume > 1.5x average
        volume_filter = vol_ma_4h[i] > 0 and volume[i] > 1.5 * vol_ma_4h[i]
        
        if position == 0:
            # Look for long entry: price breaks above 20-period high + uptrend + volume
            if close[i] > high_20[i] and close[i] > ema_50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Look for short entry: price breaks below 20-period low + downtrend + volume
            elif close[i] < low_20[i] and close[i] < ema_50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long position: exit on breakdown below 20-period low or trend reversal
            if close[i] < low_20[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short position: exit on breakout above 20-period high or trend reversal
            if close[i] > high_20[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals