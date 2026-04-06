#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian breakout with volume confirmation and 1w EMA trend filter
# Long when price breaks above 12h Donchian(20) high, volume > 2x average, and 1w EMA(50) rising
# Short when price breaks below 12h Donchian(20) low, volume > 2x average, and 1w EMA(50) falling
# Exit when price returns to Donchian midpoint or EMA trend reverses
# Designed for low trade frequency (target: 50-150 total over 4 years) to minimize fee drag
# Works in both bull and bear markets by following the 1w trend

name = "12h_donchian_vol_ema1w_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period) on 12h timeframe
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = highest_high.values
    donchian_low = lowest_low.values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_threshold = 2.0 * volume_ma.values
    
    # 1-week EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    weekly_close = df_1w['close'].values
    ema_1w = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # EMA trend direction: rising if current > previous
    ema_rising = np.zeros(n, dtype=bool)
    ema_falling = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if not np.isnan(ema_1w_aligned[i]) and not np.isnan(ema_1w_aligned[i-1]):
            ema_rising[i] = ema_1w_aligned[i] > ema_1w_aligned[i-1]
            ema_falling[i] = ema_1w_aligned[i] < ema_1w_aligned[i-1]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if required data not available
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(volume_threshold[i]) or np.isnan(ema_1w_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Exit conditions
        if position == 1:  # long position
            if close[i] <= donchian_mid[i] or not ema_rising[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            if close[i] >= donchian_mid[i] or not ema_falling[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with volume confirmation and trend filter
            # Long: breakout above Donchian high + volume + rising 1w EMA
            if (close[i] > donchian_high[i] and volume[i] > volume_threshold[i] and ema_rising[i]):
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low + volume + falling 1w EMA
            elif (close[i] < donchian_low[i] and volume[i] > volume_threshold[i] and ema_falling[i]):
                signals[i] = -0.25
                position = -1
    
    return signals