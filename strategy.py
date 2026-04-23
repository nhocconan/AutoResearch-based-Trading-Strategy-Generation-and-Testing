#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper AND 12h EMA50 rising AND volume > 1.5x average.
Short when price breaks below Donchian lower AND 12h EMA50 falling AND volume > 1.5x average.
Exit when price reverts to Donchian midline (20-period average) or volume drops below average.
Donchian channels provide structural breakout levels, 12h EMA50 filters trend direction,
volume confirmation ensures conviction. Designed for 4h timeframe targeting 75-200 total trades over 4 years.
Works in both bull and bear markets by only taking trend-aligned breakouts with volume.
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
    
    # Load 12h data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate EMA50 on 12h data
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_12h_aligned[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        upper = high_max[i]
        lower = low_min[i]
        midline = donchian_mid[i]
        
        # EMA50 slope (rising/falling) - compare to previous bar
        if i > 100:
            ema50_prev = ema50_12h_aligned[i-1]
            ema50_rising = ema50_val > ema50_prev
            ema50_falling = ema50_val < ema50_prev
        else:
            ema50_rising = False
            ema50_falling = False
        
        if position == 0:
            # Long: Break above upper AND EMA50 rising AND volume spike
            if (price > upper and ema50_rising and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower AND EMA50 falling AND volume spike
            elif (price < lower and ema50_falling and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price reverts to midline OR volume drops below average
                if (price <= midline or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price reverts to midline OR volume drops below average
                if (price >= midline or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0