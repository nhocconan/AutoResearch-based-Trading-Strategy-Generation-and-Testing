#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with 1d trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high AND price > 1d EMA(200) AND volume > 2x average
# Enter short when: price breaks below Donchian(20) low AND price < 1d EMA(200) AND volume > 2x average
# Exit when: price crosses Donchian midpoint OR opposite breakout occurs
# Uses daily trend filter to avoid counter-trend trades, targeting 100-150 trades over 4 years

name = "12h_donchian20_1dema200_vol_v1"
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
    
    # Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # 1d EMA(200) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200 = pd.Series(close_1d).ewm(span=200, adjust=False).mean().values
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200)
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midpoint OR opposite breakout with volume
            if close[i] < donchian_mid[i] or (close[i] < donchian_low[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midpoint OR opposite breakout with volume
            if close[i] > donchian_mid[i] or (close[i] > donchian_high[i] and volume[i] > volume_threshold[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + trend filter + volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema_200_aligned[i] and volume[i] > volume_threshold[i]:
                # Bullish breakout above resistance with uptrend and volume
                signals[i] = 0.25
                position = 1
            elif close[i] < donchian_low[i] and close[i] < ema_200_aligned[i] and volume[i] > volume_threshold[i]:
                # Bearish breakout below support with downtrend and volume
                signals[i] = -0.25
                position = -1
    
    return signals