#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Index (Bull Power/Bear Power) with 1d EMA50 trend filter and volume confirmation.
Long when Bull Power > 0, price > 1d EMA50, and volume > 1.5x average.
Short when Bear Power < 0, price < 1d EMA50, and volume > 1.5x average.
Exit when power reverses or price crosses 1d EMA50.
Elder Ray measures bull/bear strength relative to EMA13. Combined with 1d EMA50 trend filter,
it captures strong moves in both bull and bear markets while avoiding chop.
Designed for 6h timeframe targeting 50-150 total trades over 4 years with volume confirmation
to reduce false signals and minimize fee drag.
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
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on 1d data
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMA50 to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray Index (requires EMA13 on 6h data)
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13  # Bull Power = High - EMA13
    bear_power = low - ema_13   # Bear Power = Low - EMA13
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_1d_aligned[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        vol_ma_val = vol_ma[i]
        price = close[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND price > 1d EMA50 AND volume spike
            if (bull_val > 0 and price > ema_50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bear Power < 0 AND price < 1d EMA50 AND volume spike
            elif (bear_val < 0 and price < ema_50_val and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
                entry_price = price
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power >= 0 OR price <= 1d EMA50
                if (bull_val <= 0 or price <= ema_50_val):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power <= 0 OR price >= 1d EMA50
                if (bear_val >= 0 or price >= ema_50_val):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dEMA50_Volume"
timeframe = "6h"
leverage = 1.0