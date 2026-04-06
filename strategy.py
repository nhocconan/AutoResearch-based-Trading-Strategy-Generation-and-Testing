#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian channel breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) upper band, price > weekly EMA(20), volume > 1.5x avg
# Enter short when: price breaks below Donchian(20) lower band, price < weekly EMA(20), volume > 1.5x avg
# Exit when price returns to Donchian mid-point or opposite breakout occurs
# Uses weekly trend to filter breakouts in strong moves, targeting 50-100 trades over 4 years

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
    
    # Donchian Channel (20) on 1d
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_middle = (donchian_upper + donchian_lower) / 2
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Wait for Donchian to stabilize
        # Skip if required data not available
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian middle OR opposite breakout (below lower band)
            if close[i] < donchian_middle[i] or close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian middle OR opposite breakout (above upper band)
            if close[i] > donchian_middle[i] or close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries: Donchian break + trend filter + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_upper[i] and close[i] > ema_20_aligned[i]:
                    # Bullish breakout above weekly EMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_lower[i] and close[i] < ema_20_aligned[i]:
                    # Bearish breakout below weekly EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals