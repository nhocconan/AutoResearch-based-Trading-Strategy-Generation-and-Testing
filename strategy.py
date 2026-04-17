#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with volume confirmation and 12h EMA34 trend filter.
Long when price breaks above Donchian upper channel with volume > 1.5x 20-period average and 12h EMA34 rising.
Short when price breaks below Donchian lower channel with volume > 1.5x 20-period average and 12h EMA34 falling.
Exit on opposite Donchian channel touch or volume drying up.
Position sizing: 0.30 for entries, 0 for exits.
Target: 75-200 total trades over 4 years (19-50/year).
Works in bull markets by capturing breakouts and in bear markets by shorting breakdowns.
"""

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
    
    # Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 12h data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA34 on 12h
    ema_34 = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_rising = ema_34 > np.roll(ema_34, 1)
    ema_34_falling = ema_34 < np.roll(ema_34, 1)
    # Handle first value
    ema_34_rising[0] = False
    ema_34_falling[0] = False
    
    # Align all to 4h
    highest_high_aligned = align_htf_to_ltf(prices, pd.DataFrame({'high': high}), highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, pd.DataFrame({'low': low}), lowest_low)
    vol_ma_20_aligned = align_htf_to_ltf(prices, pd.DataFrame({'volume': volume}), vol_ma_20)
    ema_34_aligned = align_htf_to_ltf(prices, df_12h, ema_34)
    ema_34_rising_aligned = align_htf_to_ltf(prices, df_12h, ema_34_rising)
    ema_34_falling_aligned = align_htf_to_ltf(prices, df_12h, ema_34_falling)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(ema_34_aligned[i]) or
            np.isnan(ema_34_rising_aligned[i]) or np.isnan(ema_34_falling_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: breakout above upper Donchian with volume spike and rising 12h EMA34
            if (close[i] > highest_high_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                ema_34_rising_aligned[i]):
                signals[i] = 0.30
                position = 1
            # Short: breakdown below lower Donchian with volume spike and falling 12h EMA34
            elif (close[i] < lowest_low_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  ema_34_falling_aligned[i]):
                signals[i] = -0.30
                position = -1
        
        elif position == 1:
            # Exit long: price touches lower Donchian or volume dries up
            if (close[i] < lowest_low_aligned[i] or 
                volume[i] < vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        
        elif position == -1:
            # Exit short: price touches upper Donchian or volume dries up
            if (close[i] > highest_high_aligned[i] or 
                volume[i] < vol_ma_20_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

name = "4h_Donchian20_Volume_EMA34Trend"
timeframe = "4h"
leverage = 1.0