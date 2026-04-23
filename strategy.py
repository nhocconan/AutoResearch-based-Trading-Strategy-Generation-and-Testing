#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1d ATR filter and weekly EMA50 trend filter
- Uses 6h Donchian channel breakout for entry signals
- 1d ATR(14) filter: only trade when volatility is elevated (> 1.5x 50-period average)
- Weekly EMA(50) trend filter: align with higher timeframe trend (only long when price > weekly EMA)
- Volume confirmation (> 1.5x 20-period average) filters low-momentum breakouts
- Designed for 6h timeframe targeting 12-37 trades/year (50-150 over 4 years)
- Works in both bull and bear markets by combining volatility expansion with trend following
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
    
    # Calculate 6h Donchian channel (20-period)
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate daily ATR(14) for volatility filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # First period
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50, 20)  # Donchian, ATR, weekly EMA, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(high_ma[i]) or np.isnan(low_ma[i]) or
            np.isnan(atr_14[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volatility filter: ATR > 1.5x 50-period average
        atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
        volatility_filter = atr_14[i] > 1.5 * atr_ma_50[i] if not np.isnan(atr_ma_50[i]) else False
        
        # Determine breakout conditions
        price_above_upper = close[i] > high_ma[i]
        price_below_lower = close[i] < low_ma[i]
        
        # Trend filter: price > weekly EMA for long, price < weekly EMA for short
        uptrend = close[i] > ema_50_1w_aligned[i]
        downtrend = close[i] < ema_50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above upper Donchian, uptrend, volatility filter, volume spike
            long_signal = (price_above_upper and 
                          uptrend and
                          volatility_filter and
                          volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below lower Donchian, downtrend, volatility filter, volume spike
            short_signal = (price_below_lower and 
                           downtrend and
                           volatility_filter and
                           volume[i] > 1.5 * vol_ma[i])
            
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

name = "6h_Donchian20_Breakout_1dATR_VolumeFilter_1wEMA50_Trend"
timeframe = "6h"
leverage = 1.0