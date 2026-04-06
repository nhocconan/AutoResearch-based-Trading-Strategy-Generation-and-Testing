#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with 1w trend filter and volume confirmation
# Enter long when price breaks above Donchian(20) high, price > 1w EMA(20), volume > 2x average
# Enter short when price breaks below Donchian(20) low, price < 1w EMA(20), volume > 2x average
# Exit when price returns to Donchian midpoint or opposite breakout occurs
# Uses weekly trend to filter breakouts in strong moves, targeting 50-100 trades over 4 years

name = "1d_donchian20_1wema_vol_v1"
timeframe = "1d"
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
    
    # Donchian channel (20-period) on 1d
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > highest_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts: price outside Donchian bands + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if high[i] > highest_high[i] and close[i] > ema_20_aligned[i]:
                    # Bullish breakout with weekly uptrend
                    signals[i] = 0.25
                    position = 1
                elif low[i] < lowest_low[i] and close[i] < ema_20_aligned[i]:
                    # Bearish breakout with weekly downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals