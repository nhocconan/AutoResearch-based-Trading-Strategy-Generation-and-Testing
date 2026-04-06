#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA(150) trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1d EMA(150), volume > 1.8x avg
# Enter short when: price breaks below Donchian(20) low, price < 1d EMA(150), volume > 1.8x avg
# Exit when: price retraces to midpoint of Donchian channel OR opposite breakout occurs
# Uses daily trend filter to avoid counter-trend trades, targeting 50-150 trades over 4 years
# 12h timeframe balances signal frequency with cost efficiency in bear/bull markets

name = "12h_donchian20_1dema150_vol_v1"
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
    
    # Donchian channel (20-period) on 12h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2
    
    # 1d EMA(150) for trend filter (longer-term trend)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_150 = pd.Series(close_1d).ewm(span=150, adjust=False).mean().values
    ema_150_aligned = align_htf_to_ltf(prices, df_1d, ema_150)
    
    # Volume confirmation: volume > 1.8x 20-period average (stricter filter)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_150_aligned[i]) or np.isnan(volume_threshold[i])):
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
                if close[i] > high_20[i] and close[i] > ema_150_aligned[i]:
                    # Bullish breakout above Donchian high with daily uptrend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and close[i] < ema_150_aligned[i]:
                    # Bearish breakdown below Donchian low with daily downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals