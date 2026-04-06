#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high + price > 1d EMA(50) + volume > 1.5x avg
# Short when price breaks below Donchian(20) low + price < 1d EMA(50) + volume > 1.5x avg
# Uses 1d EMA for trend filter to avoid counter-trend trades
# Targets 75-200 total trades over 4 years by requiring trend alignment and volume spike

name = "4h_donchian_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price closes below Donchian low or reverses against EMA trend
            if close[i] < low_roll[i] or close[i] < ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above Donchian high or reverses against EMA trend
            if close[i] > high_roll[i] or close[i] > ema_50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend and volume confirmation
            if close[i] > high_roll[i] and close[i] > ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = 0.25
                position = 1
            elif close[i] < low_roll[i] and close[i] < ema_50_aligned[i] and volume[i] > volume_threshold[i]:
                signals[i] = -0.25
                position = -1
    
    return signals