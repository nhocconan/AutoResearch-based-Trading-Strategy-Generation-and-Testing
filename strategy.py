#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1d EMA34 AND volume > 1.5x average.
Short when price breaks below Donchian(20) low AND price < 1d EMA34 AND volume > 1.5x average.
Exit when price crosses 1d EMA34 in opposite direction or volume drops below average.
Donchian channels provide structural breakout levels, 1d EMA34 ensures higher timeframe trend alignment,
volume confirmation filters weak breakouts. Designed for 12h timeframe targeting 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for EMA34 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA34 on 1d data
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 12h timeframe
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian(20) on 12h data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or i < lookback):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema34_val = ema34_1d_aligned[i]
        price = close[i]
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().iloc[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: break above Donchian high AND price > 1d EMA34 AND volume spike
            if (price > highest_high[i] and price > ema34_val and vol_current > 1.5 * vol_ma):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low AND price < 1d EMA34 AND volume spike
            elif (price < lowest_low[i] and price < ema34_val and vol_current > 1.5 * vol_ma):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below 1d EMA34 OR volume drops below average
                if (price < ema34_val or vol_current < vol_ma):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above 1d EMA34 OR volume drops below average
                if (price > ema34_val or vol_current < vol_ma):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1dEMA34_Volume"
timeframe = "12h"
leverage = 1.0