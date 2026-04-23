#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1w EMA50 trend filter + volume confirmation.
Long when price breaks above upper Donchian channel AND price > 1w EMA50 AND volume > 1.5x average.
Short when price breaks below lower Donchian channel AND price < 1w EMA50 AND volume > 1.5x average.
Exit when price returns to middle of Donchian channel (mean reversion) or volume drops below average.
Designed for 12h timeframe targeting 50-150 total trades over 4 years with low frequency to minimize fee drag.
Uses weekly trend filter to avoid counter-trend trades in bear markets (2022 crash, 2025 sideways).
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
    
    # Load 1w data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w data
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 12h timeframe
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate Donchian channels on 12h data (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_ma + low_ma) / 2.0
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(high_ma[i]) or 
            np.isnan(low_ma[i]) or np.isnan(donchian_mid[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_1w_aligned[i]
        upper_channel = high_ma[i]
        lower_channel = low_ma[i]
        mid_channel = donchian_mid[i]
        price = close[i]
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Break above upper Donchian AND price > 1w EMA50 AND volume spike
            if (price > upper_channel and price > ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Break below lower Donchian AND price < 1w EMA50 AND volume spike
            elif (price < lower_channel and price < ema50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price returns to middle of channel OR volume drops below average
                if (price <= mid_channel or vol_current < vol_ma_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price returns to middle of channel OR volume drops below average
                if (price >= mid_channel or vol_current < vol_ma_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_1wEMA50_Volume"
timeframe = "12h"
leverage = 1.0