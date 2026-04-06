#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily Donchian channel breakout with weekly trend filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high, weekly EMA(20) is rising, volume > 1.5x average
# Enter short when: price breaks below Donchian(20) low, weekly EMA(20) is falling, volume > 1.5x average
# Exit when: price returns to Donchian midpoint or opposite breakout occurs
# Uses weekly trend to filter breakouts in strong trends, targeting 50-100 trades over 4 years

name = "1d_donchian20_1w_ema_vol_v1"
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
    
    # Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Weekly EMA(20) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_20 = pd.Series(close_1w).ewm(span=20, adjust=False).mean().values
    ema_20_prev = np.roll(ema_20, 1)
    ema_20_prev[0] = ema_20[0]
    ema_rising = ema_20 > ema_20_prev
    ema_falling = ema_20 < ema_20_prev
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20)
    ema_rising_aligned = align_htf_to_ltf(prices, df_1w, ema_rising)
    ema_falling_aligned = align_htf_to_ltf(prices, df_1w, ema_falling)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price returns to midpoint OR opposite breakout
            if close[i] <= donchian_mid[i] or close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price returns to midpoint OR opposite breakout
            if close[i] >= donchian_mid[i] or close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with weekly trend and volume
            if volume[i] > volume_threshold[i]:
                if close[i] > donchian_high[i] and ema_20_aligned[i] > ema_20[max(0, i-1)]:
                    # Bullish breakout with rising weekly EMA
                    signals[i] = 0.25
                    position = 1
                elif close[i] < donchian_low[i] and ema_20_aligned[i] < ema_20[max(0, i-1)]:
                    # Bearish breakout with falling weekly EMA
                    signals[i] = -0.25
                    position = -1
    
    return signals