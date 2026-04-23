#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout (20) with 1d EMA(50) trend filter and volume spike confirmation.
- Donchian(20) provides robust breakout levels that work in trending and ranging markets
- 1d EMA(50) filters for higher-timeframe trend direction to avoid counter-trend whipsaws
- Volume confirmation (> 2.0x 20-period average) ensures breakout has momentum
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
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
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate Donchian channels (20-period) for breakout levels
    # Upper = max(high, 20), Lower = min(low, 20)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        # Long: price breaks above upper Donchian with volume and uptrend
        # Short: price breaks below lower Donchian with volume and downtrend
        price_above_upper = close[i] > high_roll[i]
        price_below_lower = close[i] < low_roll[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper, uptrend, volume spike
            long_signal = (price_above_upper and 
                          uptrend and
                          volume[i] > 2.0 * vol_ma[i])
            
            # Short conditions: price breaks below lower, downtrend, volume spike
            short_signal = (price_below_lower and 
                           downtrend and
                           volume[i] > 2.0 * vol_ma[i])
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite break or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below lower Donchian or trend turns down
                if (price_below_lower or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above upper Donchian or trend turns up
                if (price_above_upper or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0