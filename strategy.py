#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, weekly close > weekly open (bullish weekly candle), volume > 1.5x 20-period average
# Enter short when: price breaks below Donchian(20) low, weekly close < weekly open (bearish weekly candle), volume > 1.5x 20-period average
# Exit when: price crosses back through Donchian(20) midline OR opposite breakout occurs
# Uses weekly trend to filter false breakouts, targeting 75-150 trades over 4 years

name = "6h_donchian20_weeklytrend_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) channels
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # Weekly trend filter: bullish if weekly close > weekly open
    df_1w = get_htf_data(prices, '1w')
    weekly_open = df_1w['open'].values
    weekly_close = df_1w['close'].values
    weekly_bullish = weekly_close > weekly_open
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(volume_threshold[i]) or np.isnan(weekly_bullish_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midline OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midline OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with weekly trend filter and volume
            if volume[i] > volume_threshold[i]:
                if high[i] > high_roll[i] and weekly_bullish_aligned[i]:
                    # Bullish breakout with bullish weekly candle
                    signals[i] = 0.25
                    position = 1
                elif low[i] < low_roll[i] and not weekly_bullish_aligned[i]:
                    # Bearish breakout with bearish weekly candle
                    signals[i] = -0.25
                    position = -1
    
    return signals