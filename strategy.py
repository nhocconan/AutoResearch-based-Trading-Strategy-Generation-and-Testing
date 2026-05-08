#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with 12h trend filter and volume confirmation
# Uses Donchian channels for trend-following breakouts, filtered by 12h EMA trend direction
# and volume spikes to avoid false breakouts. Designed for low-frequency trading (<100 trades/year)
# to minimize fee drag while capturing major trend moves in both bull and bear markets.

name = "4h_Donchian20_12hEMA25_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema25_12h = pd.Series(close_12h).ewm(span=25, adjust=False, min_periods=25).mean().values
    ema25_12h_aligned = align_htf_to_ltf(prices, df_12h, ema25_12h)
    
    # Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (1.5x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure Donchian and EMA have enough data
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_max[i]) or np.isnan(low_min[i]) or 
            np.isnan(ema25_12h_aligned[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper with 12h uptrend and volume spike
            if (close[i] > high_max[i] and 
                close[i] > ema25_12h_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower with 12h downtrend and volume spike
            elif (close[i] < low_min[i] and 
                  close[i] < ema25_12h_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower or trend fails
            if (close[i] < low_min[i] or 
                close[i] < ema25_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper or trend fails
            if (close[i] > high_max[i] or 
                close[i] > ema25_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals