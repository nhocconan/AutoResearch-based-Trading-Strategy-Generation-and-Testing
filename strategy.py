#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using 12h Supertrend for trend direction and 4h Donchian breakout with volume confirmation.
# In strong trends (Supertrend direction), trade breakouts in the direction of the trend.
# In weak trends or ranging markets, avoid trades to reduce whipsaw.
# Volume > 1.5x 20-period average confirms breakout strength.
# Target: 20-40 trades/year per symbol (80-160 total over 4 years) to minimize fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    
    # 12h Supertrend (ATR=10, mult=3) for trend direction
    atr_len = 10
    mult = 3
    if len(df_12h) < atr_len:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])
    
    # ATR
    atr = pd.Series(tr).ewm(span=atr_len, adjust=False, min_periods=atr_len).mean().values
    
    # Upper and Lower Bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + mult * atr
    lower_band = hl2 - mult * atr
    
    # Supertrend calculation
    supertrend = np.zeros_like(close_12h)
    direction = np.ones_like(close_12h)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0] if not np.isnan(upper_band[0]) else close_12h[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(atr[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
        else:
            if close_12h[i] > supertrend[i-1]:
                direction[i] = 1
            elif close_12h[i] < supertrend[i-1]:
                direction[i] = -1
            else:
                direction[i] = direction[i-1]
            
            if direction[i] == 1:
                supertrend[i] = max(lower_band[i], supertrend[i-1])
            else:
                supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend direction to 4h timeframe
    supertrend_direction_aligned = align_htf_to_ltf(prices, df_12h, direction)
    
    # 4h Donchian Channel (20 periods)
    donch_len = 20
    if len(high) < donch_len:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=donch_len, min_periods=donch_len).max().values
    lowest_low = pd.Series(low).rolling(window=donch_len, min_periods=donch_len).min().values
    
    # Volume confirmation: 1.5x average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(donch_len, 20)
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(supertrend_direction_aligned[i]) or 
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from 12h Supertrend
        uptrend = supertrend_direction_aligned[i] == 1
        downtrend = supertrend_direction_aligned[i] == -1
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            if uptrend and volume_confirmed:
                # Long breakout above Donchian high in uptrend
                if close[i] > highest_high[i]:
                    position = 1
                    signals[i] = position_size
                else:
                    signals[i] = 0.0
            elif downtrend and volume_confirmed:
                # Short breakdown below Donchian low in downtrend
                if close[i] < lowest_low[i]:
                    position = -1
                    signals[i] = -position_size
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price breaks below Donchian low or trend reverses
            if close[i] < lowest_low[i] or supertrend_direction_aligned[i] != 1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price breaks above Donchian high or trend reverses
            if close[i] > highest_high[i] or supertrend_direction_aligned[i] != -1:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_Supertrend_Donchian_Breakout_v1"
timeframe = "4h"
leverage = 1.0