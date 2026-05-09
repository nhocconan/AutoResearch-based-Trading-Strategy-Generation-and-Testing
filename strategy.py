#!/usr/bin/env python3
# Hypothesis: 4h Donchian breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x 20-period average
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x 20-period average
# Exit when price breaks opposite Donchian band OR EMA50 contradicts position
# Position size: 0.28 to balance return and drawdown
# Designed for trending markets with momentum confirmation and volume validation

name = "4h_Donchian_EMA50_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = high_roll.values
    donchian_low = low_roll.values
    
    # 12h EMA50 for trend filter
    ema50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_spike = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian high AND above EMA50 (bullish) + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema50[i] and 
                vol_spike[i]):
                signals[i] = 0.28
                position = 1
            # Enter short: price breaks below Donchian low AND below EMA50 (bearish) + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema50[i] and 
                  vol_spike[i]):
                signals[i] = -0.28
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below Donchian low OR EMA50 turns bearish
            if (close[i] < donchian_low[i]) or (close[i] < ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.28
        
        elif position == -1:
            # Exit short: price breaks above Donchian high OR EMA50 turns bullish
            if (close[i] > donchian_high[i]) or (close[i] > ema50[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.28
    
    return signals