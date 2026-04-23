#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR filter and volume confirmation
- Uses Donchian channel (20-period high/low) for breakout signals on 4h
- 1d ATR(14) measures volatility - only trade when ATR > 20-period average (avoid low vol chop)
- Volume confirmation (> 1.3x 20-period average) ensures momentum behind breakout
- Position size 0.25 for good risk/reward
- Works in both bull and bear markets by trading breakouts in direction of 4h trend (EMA50)
- Designed for ~30-50 trades/year to minimize fee drag
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
    
    # Calculate 4h Donchian channel (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA(50) for trend filter
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1d ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # 20-period average of ATR for volatility regime filter
    atr_ma = pd.Series(atr_14).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_aligned = align_htf_to_ltf(prices, df_1d, atr_ma)
    
    # Volume confirmation: > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 20)  # Donchian, EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(ema_50[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ma_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        breakout_up = close[i] > high_ma[i]
        breakout_down = close[i] < low_ma[i]
        
        # Trend filter: price > EMA50 for uptrend, price < EMA50 for downtrend
        uptrend = close[i] > ema_50[i]
        downtrend = close[i] < ema_50[i]
        
        # Volatility filter: only trade when ATR > MA (avoid low volatility chop)
        high_volatility = atr_14_aligned[i] > atr_ma_aligned[i]
        
        if position == 0:
            # Long conditions: bullish breakout, uptrend, high volume, high volatility
            long_signal = (breakout_up and 
                          uptrend and
                          volume[i] > 1.3 * vol_ma[i] and
                          high_volatility)
            
            # Short conditions: bearish breakout, downtrend, high volume, high volatility
            short_signal = (breakout_down and 
                           downtrend and
                           volume[i] > 1.3 * vol_ma[i] and
                           high_volatility)
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price falls below Donchian low or trend turns down
                if (breakout_down or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend turns up
                if (breakout_up or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_4hEMA50_Trend_VolumeATRFilter"
timeframe = "4h"
leverage = 1.0