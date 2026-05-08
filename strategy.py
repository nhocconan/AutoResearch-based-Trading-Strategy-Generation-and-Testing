#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and EMA trend filter.
# Breakouts from 20-period Donchian channels capture momentum in trending markets.
# Volume confirmation ensures breakout validity. EMA filter aligns with trend direction.
# Designed for low trade frequency (20-40/year) to minimize fee drag and maximize edge.
# Works in both bull and bear markets by following the trend (long in uptrend, short in downtrend).

name = "4h_Donchian20_Breakout_Volume_Trend"
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
    
    # Donchian channel (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # EMA(50) for trend filter
    close_series = pd.Series(close)
    ema_50 = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period EMA of volume
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_confirm = volume > (vol_ema * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for EMA(50)
    
    for i in range(start_idx, n):
        if position == 0:
            # Long: break above Donchian high with volume and uptrend
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50[i] and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and downtrend
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50[i] and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: break below Donchian low or trend turns down
            if close[i] < donchian_low[i] or close[i] < ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: break above Donchian high or trend turns up
            if close[i] > donchian_high[i] or close[i] > ema_50[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals