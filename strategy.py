#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation
# Enter long when price breaks above 20-day Donchian high AND price > weekly EMA(21) AND volume > 1.5x average
# Enter short when price breaks below 20-day Donchian low AND price < weekly EMA(21) AND volume > 1.5x average
# Exit when price crosses the Donchian midpoint or opposite breakout occurs
# Uses weekly trend to filter breakouts in strong trends, targeting 50-100 trades over 4 years

name = "1d_donchian20_weeklyema_vol_v1"
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
    
    # 20-period Donchian channels (daily)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly EMA(21) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_21 = pd.Series(close_1w).ewm(span=21, adjust=False).mean().values
    ema_21_aligned = align_htf_to_ltf(prices, df_1w, ema_21)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_21_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price crosses below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or low[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price crosses above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or high[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: Donchian breakout + weekly trend filter + volume
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above Donchian high AND above weekly EMA
                if close[i] > donchian_high[i] and close[i] > ema_21_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND below weekly EMA
                elif close[i] < donchian_low[i] and close[i] < ema_21_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals