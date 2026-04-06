#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when price breaks above Donchian high + 1d close > 1d open (bullish day) + volume > 1.5x average
# Short when price breaks below Donchian low + 1d close < 1d open (bearish day) + volume > 1.5x average
# Exit when price crosses Donchian midpoint
# Uses 4h timeframe targeting 75-200 total trades over 4 years (19-50/year)
# Works in trending markets by following breakouts with daily trend filter

name = "4h_donchian_1d_trend_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mid = (donch_high + donch_low) / 2
    
    # 1d trend: bullish day = close > open
    df_1d = get_htf_data(prices, '1d')
    daily_bullish = (df_1d['close'].values > df_1d['open'].values)
    daily_bullish_aligned = align_htf_to_ltf(prices, df_1d, daily_bullish)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(daily_bullish_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses Donchian midpoint
        if position == 1:  # long position
            if close[i] <= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donch_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Bullish breakout: price above Donchian high + bullish day + volume
            if (close[i] > donch_high[i] and 
                daily_bullish_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Bearish breakout: price below Donchian low + bearish day + volume
            elif (close[i] < donch_low[i] and 
                  not daily_bullish_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals