# A robust 4-hour momentum strategy combining Donchian breakouts, volume confirmation, and ADX trend filtering. 
# Designed to work in both bull and bear markets by focusing on breakout strength and institutional participation.
# The strategy avoids false breakouts by requiring volume confirmation and trend alignment.
# Target: 20-40 trades per year with clear entry/exit rules to minimize fee drag.

#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 4h Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ADX(14) for trend strength on 4h data
    tr1 = np.maximum(high[1:], low[:-1]) - np.minimum(high[1:], low[:-1])
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    plus_dm = np.where((high[1:] - high[:-1]) > (low[:-1] - low[1:]), 
                       np.maximum(high[1:] - high[:-1], 0), 0)
    minus_dm = np.where((low[:-1] - low[1:]) > (high[1:] - high[:-1]), 
                        np.maximum(low[:-1] - low[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / (atr + 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(20, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(avg_volume[i]) or np.isnan(adx[i])):
            continue
        
        # Long conditions: price breaks above Donchian high with volume confirmation and ADX > 20
        if (close[i] > highest_high[i] and 
            volume[i] > avg_volume[i] * 1.5 and 
            adx[i] > 20 and 
            position <= 0):
            position = 1
            signals[i] = position_size
        # Short conditions: price breaks below Donchian low with volume confirmation and ADX > 20
        elif (close[i] < lowest_low[i] and 
              volume[i] > avg_volume[i] * 1.5 and 
              adx[i] > 20 and 
              position >= 0):
            position = -1
            signals[i] = -position_size
        # Exit conditions: reverse signal or loss of momentum (ADX < 20)
        elif position == 1 and (close[i] < lowest_low[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > highest_high[i] or adx[i] < 20):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Donchian_Volume_ADX_Momentum"
timeframe = "4h"
leverage = 1.0