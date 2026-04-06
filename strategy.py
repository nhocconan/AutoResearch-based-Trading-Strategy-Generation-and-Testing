#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1d EMA200 trend + volume confirmation
# Long when price breaks above Donchian high (20) AND close > 1d EMA200 AND volume > 1.5x average
# Short when price breaks below Donchian low (20) AND close < 1d EMA200 AND volume > 1.5x average
# Exit when price crosses back through Donchian midpoint
# Uses 12h timeframe to reduce trade frequency, targets 50-150 total trades over 4 years
# Works in bull markets via trend-following, works in bear via short signals
# Volume confirmation reduces false breakouts, EMA200 filter ensures trend alignment

name = "12h_donchian20_1d_ema200_vol_v1"
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
    
    # Donchian Channel (20-period) on 12h
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    daily_close = df_1d['close'].values
    
    ema_200 = pd.Series(daily_close).ewm(span=200, min_periods=200, adjust=False).mean()
    ema_200_values = ema_200.values
    
    # Align daily EMA200 to 12h timeframe
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_values)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 1.5 * volume_ma.values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or \
           np.isnan(ema_200_aligned[i]) or np.isnan(volume_threshold[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions: price crosses back through Donchian midpoint
        if position == 1:  # long position
            if close[i] <= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with trend and volume confirmation
            # Long: price breaks above Donchian high AND above 1d EMA200 + volume
            if (close[i] > donchian_high[i] and 
                close[i] > ema_200_aligned[i] and 
                volume[i] > volume_threshold[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1d EMA200 + volume
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_200_aligned[i] and 
                  volume[i] > volume_threshold[i]):
                signals[i] = -0.25
                position = -1
    
    return signals