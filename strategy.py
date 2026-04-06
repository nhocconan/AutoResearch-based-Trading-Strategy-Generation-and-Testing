#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian breakout with 1w trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, price > 1w EMA(20), volume > 1.8x avg
# Enter short when: price breaks below Donchian(20) low, price < 1w EMA(20), volume > 1.8x avg
# Exit when price crosses Donchian midpoint OR opposite breakout occurs
# Focus on strong trending moves with volume confirmation to avoid chop, targeting 75-150 trades over 4 years
# Works in bull (catches breakouts) and bear (short breakdowns) with trend filter to avoid counter-trend trades

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
    
    # Donchian channel (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2.0
    
    # 1w EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.8x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.8 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
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
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if volume[i] > volume_threshold[i]:
                if high[i] > high_roll[i] and close[i] > ema_20_aligned[i]:
                    # Bullish breakout above Donchian high with uptrend filter
                    signals[i] = 0.25
                    position = 1
                elif low[i] < low_roll[i] and close[i] < ema_20_aligned[i]:
                    # Bearish breakdown below Donchian low with downtrend filter
                    signals[i] = -0.25
                    position = -1
    
    return signals