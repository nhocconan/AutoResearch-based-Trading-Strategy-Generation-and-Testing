#!/usr/bin/env python3
"""
Hypothesis: 1-day donchian breakout with 1-week ema trend filter and volume confirmation.
Long when price breaks above 20-period donchian upper, price > 1-week ema50, volume > 1.5x average.
Short when price breaks below 20-period donchian lower, price < 1-week ema50, volume > 1.5x average.
Exit when price returns to donchian middle or trend weakens.
Designed for low trade frequency (~10-25/year) to capture strong trends while minimizing whipsaws.
Works in both bull and bear markets by requiring strong trend confirmation (price vs EMA).
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
    
    # Load 1-week data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1-week EMA50
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 1-day Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Volume average (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        middle = mid_20[i]
        ema50 = ema50_1w_aligned[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        
        if position == 0:
            # Long: price breaks above upper band, above weekly EMA50, volume confirmation
            if (price > upper and price > ema50 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, below weekly EMA50, volume confirmation
            elif (price < lower and price < ema50 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle OR price below weekly EMA50
                if price < middle or price < ema50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle OR price above weekly EMA50
                if price > middle or price > ema50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_1wEMA50_Volume_Trend"
timeframe = "1d"
leverage = 1.0