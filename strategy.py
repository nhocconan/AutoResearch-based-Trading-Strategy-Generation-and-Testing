#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d trend filter and volume confirmation
# Long when: price breaks above Donchian upper, price > 1d EMA(100), volume > 1.5x avg
# Short when: price breaks below Donchian lower, price < 1d EMA(100), volume > 1.5x avg
# Exit when: price crosses Donchian midline or opposite breakout occurs
# Uses daily EMA to align with higher timeframe trend, targeting 50-150 trades over 4 years

name = "12h_donchian20_1dema_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian(20) on 12h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(100) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_100 = pd.Series(close_1d).ewm(span=100, adjust=False).mean().values
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below midline OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above midline OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if high[i] > donchian_high[i] and close[i] > ema_100_aligned[i]:
                    # Bullish breakout above upper band with daily uptrend
                    signals[i] = 0.25
                    position = 1
                elif low[i] < donchian_low[i] and close[i] < ema_100_aligned[i]:
                    # Bearish breakout below lower band with daily downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals