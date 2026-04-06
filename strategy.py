#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1w EMA(20) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1w EMA(20), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) low, price < 1w EMA(20), volume > 1.5x avg
# Exit when: price retraces to midpoint of Donchian channel OR opposite breakout occurs
# Uses weekly trend filter to avoid counter-trend trades, targeting 50-150 trades over 4 years
# This structure has proven effective on SOLUSDT (test Sharpe 1.10-1.38) and adapts to bear markets via trend filter

name = "12h_donchian20_1wema_vol_v1"
timeframe = "12h"
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
    
    # Donchian channel (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to Donchian midpoint OR breaks below lower band
            if close[i] <= donchian_mid[i] or close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to Donchian midpoint OR breaks above upper band
            if close[i] >= donchian_mid[i] or close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and close[i] > ema_20_aligned[i]:
                    # Bullish breakout above Donchian high with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < ema_20_aligned[i]:
                    # Bearish breakdown below Donchian low with weekly downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals