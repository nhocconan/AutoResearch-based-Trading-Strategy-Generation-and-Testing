#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + daily direction filter + volume confirmation
# Long when price breaks above Donchian high(20) AND daily close > daily open AND volume > 1.5x avg
# Short when price breaks below Donchian low(20) AND daily close < daily open AND volume > 1.5x avg
# Exit when price crosses Donchian midline (10-period average) or opposite breakout occurs
# Uses 6h timeframe to balance trade frequency and responsiveness, targets 50-150 total trades over 4 years
# Works in bull markets via breakouts and in bear markets via short breakdowns
# Volume confirmation reduces false breakouts; daily filter ensures alignment with higher timeframe trend

name = "6h_donchian20_1d_dir_vol_v1"
timeframe = "6h"
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
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Daily direction filter: 1 if bullish (close > open), -1 if bearish (close < open)
    df_1d = get_htf_data(prices, '1d')
    daily_open = df_1d['open'].values
    daily_close = df_1d['close'].values
    daily_direction = np.where(daily_close > daily_open, 1, np.where(daily_close < daily_open, -1, 0))
    daily_direction_aligned = align_htf_to_ltf(prices, df_1d, daily_direction)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(daily_direction_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses midline OR opposite breakout occurs
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or low[i] <= donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or high[i] >= donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with daily direction alignment and volume confirmation
            # Long breakout: price > Donchian high AND daily bullish AND volume confirmation
            if (high[i] > donchian_high[i] and daily_direction_aligned[i] == 1 and volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short breakdown: price < Donchian low AND daily bearish AND volume confirmation
            elif (low[i] < donchian_low[i] and daily_direction_aligned[i] == -1 and volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals