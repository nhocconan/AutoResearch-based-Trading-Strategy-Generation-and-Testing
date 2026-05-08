#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA100 trend filter and volume confirmation.
# Long when price breaks above upper band AND price > EMA100(1d) AND volume > 1.5x 20-period average.
# Short when price breaks below lower band AND price < EMA100(1d) AND volume > 1.5x 20-period average.
# Exit when price crosses back below upper band (long) or above lower band (short).
# Uses 12h timeframe with daily trend filter to capture sustained moves while avoiding whipsaws.
# Volume filter ensures institutional participation. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_1dEMA100_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    # 1d data for EMA100 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # EMA100 on 1d close
    ema_100 = pd.Series(df_1d['close'].values).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align 1d EMA100 to 12h timeframe
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # 12h Donchian(20) channels
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for EMA100
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, price > EMA100, volume filter
            long_cond = (close[i] > donchian_upper[i]) and (close[i] > ema_100_aligned[i]) and volume_filter[i]
            # Short conditions: break below lower band, price < EMA100, volume filter
            short_cond = (close[i] < donchian_lower[i]) and (close[i] < ema_100_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: cross below upper band
            if close[i] < donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: cross above lower band
            if close[i] > donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals