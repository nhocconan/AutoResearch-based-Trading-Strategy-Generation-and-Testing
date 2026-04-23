#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above Donchian upper band (20-period) on 4h, 12h EMA50 > EMA200 (uptrend), and volume > 1.5x average.
Short when price breaks below Donchian lower band (20-period) on 4h, 12h EMA50 < EMA200 (downtrend), and volume > 1.5x average.
Exit when price returns to Donchian middle (10-period average) or volume drops below average.
Designed for low trade frequency (~25-40/year) to capture breakouts in trending markets while avoiding chop.
Works in both bull and bear markets by requiring strong trend confirmation on 12h timeframe.
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
    
    # Load 12-hour data for EMA trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12-hour EMA50 and EMA200 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Calculate Donchian channels on 4h (20-period upper/lower, 10-period middle)
    # Upper band: highest high of last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low of last 20 periods
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Middle: average of upper and lower
    middle = (highest_high + lowest_low) / 2
    
    # Volume average (20-period) on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(ema200_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50 = ema50_12h_aligned[i]
        ema200 = ema200_12h_aligned[i]
        upper = highest_high[i]
        lower = lowest_low[i]
        mid = middle[i]
        vol_ma_val = vol_ma[i]
        vol_current = volume[i]
        price = close[i]
        
        if position == 0:
            # Long: price breaks above upper band, uptrend on 12h (EMA50 > EMA200), volume confirmation
            if (price > upper and ema50 > ema200 and vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band, downtrend on 12h (EMA50 < EMA200), volume confirmation
            elif (price < lower and ema50 < ema200 and vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: price returns to middle OR volume drops below average
                if price <= mid or vol_current < vol_ma_val:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price returns to middle OR volume drops below average
                if price >= mid or vol_current < vol_ma_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_12hEMA_Trend_Volume"
timeframe = "4h"
leverage = 1.0