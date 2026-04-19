#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout (20) with 1w EMA trend filter and volume confirmation
# Long when price breaks above 10-day high AND weekly EMA(9) is rising AND volume > 1.5x average
# Short when price breaks below 10-day low AND weekly EMA(9) is falling AND volume > 1.5x average
# Exit when price returns to 10-day opposite level or trend reverses
# Designed to capture sustained moves with trend and volume confirmation
# Target: 15-25 trades/year to minimize fee drag
name = "1d_Donchian_1wEMA_Volume_v2"
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Weekly EMA(9) for trend direction
    ema_1w = pd.Series(df_1w['close'].values).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_1w_rising = ema_1w > np.roll(ema_1w, 1)  # Rising if current > previous
    ema_1w_falling = ema_1w < np.roll(ema_1w, 1)  # Falling if current < previous
    ema_1w_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_rising)
    ema_1w_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_falling)
    
    # 10-day high/low for entry (using daily high/low)
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_avg
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Ensure enough data for volume average
    
    for i in range(start_idx, n):
        if np.isnan(high_10[i]) or np.isnan(low_10[i]) or np.isnan(ema_1w_rising_aligned[i]) or np.isnan(ema_1w_falling_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long entry: break above 10-day high + weekly EMA rising + volume
            if close[i] > high_10[i] and ema_1w_rising_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: break below 10-day low + weekly EMA falling + volume
            elif close[i] < low_10[i] and ema_1w_falling_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: return to 10-day low OR weekly EMA turns falling
            if close[i] < low_10[i] or not ema_1w_rising_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: return to 10-day high OR weekly EMA turns rising
            if close[i] > high_10[i] or not ema_1w_falling_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals