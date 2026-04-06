#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w EMA(50) trend filter and volume confirmation
# Enter long when: price breaks above Donchian upper band (20-day high) AND price > 1w EMA(50) AND volume > 2x 20-day average
# Enter short when: price breaks below Donchian lower band (20-day low) AND price < 1w EMA(50) AND volume > 2x 20-day average
# Exit when: price crosses back through Donchian midpoint (10-day average of high/low) OR opposite breakout occurs
# Uses weekly trend to filter breakouts in strong moves, targeting 50-100 trades over 4 years

name = "1d_donchian20_1wema_vol_breakout_v1"
timeframe = "1d"
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # 1w EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: price breaks Donchian band + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if high[i] > high_roll[i] and close[i] > ema_50_aligned[i]:
                    # Bullish breakout above upper band with uptrend
                    signals[i] = 0.25
                    position = 1
                elif low[i] < low_roll[i] and close[i] < ema_50_aligned[i]:
                    # Bearish breakout below lower band with downtrend
                    signals[i] = -0.25
                    position = -1
    
    return signals