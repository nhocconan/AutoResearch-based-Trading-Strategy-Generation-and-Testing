#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume > 1.3x average.
Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume > 1.3x average.
Exit when price crosses the Donchian midpoint OR volume drops below average.
Donchian channels provide clear breakout levels with built-in volatility adjustment.
1d EMA50 ensures trading in direction of higher timeframe trend.
Volume confirmation avoids low-conviction breakouts.
Designed for 4h timeframe targeting 75-200 total trades over 4 years with optimal frequency to minimize fee drag.
Works in both bull and bear markets by only taking trades aligned with 1d trend.
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
    
    # Load 1d data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 4h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate Donchian(20) on 4h data
    high_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_max + low_min) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max[i]) or 
            np.isnan(low_min[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1d_aligned[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: price breaks above Donchian high AND price > 1d EMA50 AND volume spike
            if (price > high_max[i] and price > ema50_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND price < 1d EMA50 AND volume spike
            elif (price < low_min[i] and price < ema50_val and vol_current > 1.3 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses Donchian midpoint OR volume drops below average
                if (price < donchian_mid[i] or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses Donchian midpoint OR volume drops below average
                if (price > donchian_mid[i] or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_1dEMA50_Volume"
timeframe = "4h"
leverage = 1.0