#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND close > 1d EMA(50) AND volume > 1.5x average
# Short when price breaks below Donchian(20) low AND close < 1d EMA(50) AND volume > 1.5x average
# Exit when price crosses back below Donchian(20) low (long) or above Donchian(20) high (short)
# Uses 12h timeframe to reduce trade frequency, targets 75-250 total trades over 4 years
# Works in trending markets by following breakouts with trend filter

name = "12h_donchian_1d_ema_vol_v2"
timeframe = "12h"
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
    
    # Donchian Channel (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # EMA (50-period) from 1d timeframe for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    ema_50 = pd.Series(daily_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(volume_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout with trend filter and volume confirmation
            # Long: price breaks above Donchian high AND close > 1d EMA(50) AND volume > 1.5x average
            if (close[i] > highest_high[i] and close[i] > ema_50_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND close < 1d EMA(50) AND volume > 1.5x average
            elif (close[i] < lowest_low[i] and close[i] < ema_50_aligned[i] and volume[i] > 1.5 * volume_ma[i]):
                signals[i] = -0.25
                position = -1
    
    return signals