#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and ATR(14) volatility filter
- Uses Donchian channel (20-period high/low) for breakout signals
- 1d EMA(50) defines trend direction (only long when price > EMA, short when price < EMA)
- ATR(14) > 1.2x 50-period average ensures sufficient volatility for breakout follow-through
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in both bull and bear markets by trading with the 1d trend
- ATR filter reduces false breakouts during low volatility consolidation
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
    
    # Calculate daily Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Donchian channel: 20-period high/low
    high_ma = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_ma)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_ma)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate ATR(14) for volatility filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period TR is just high-low
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_ma_50 = pd.Series(atr_14).rolling(window=50, min_periods=50).mean().values
    atr_14_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    atr_ma_50_aligned = align_htf_to_ltf(prices, df_1d, atr_ma_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50, 14, 50)  # Donchian, EMA, ATR, ATR MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(atr_14_aligned[i]) or
            np.isnan(atr_ma_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine breakout conditions
        price_above_donchian_high = close[i] > donchian_high[i]
        price_below_donchian_low = close[i] < donchian_low[i]
        
        # Trend filter: price > EMA for long, price < EMA for short
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volatility filter: ATR > 1.2x ATR MA (sufficient volatility for breakout)
        vol_filter = atr_14_aligned[i] > 1.2 * atr_ma_50_aligned[i]
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, uptrend, sufficient volatility
            long_signal = (price_above_donchian_high and 
                          uptrend and
                          vol_filter)
            
            # Short conditions: price breaks below Donchian low, downtrend, sufficient volatility
            short_signal = (price_below_donchian_low and 
                           downtrend and
                           vol_filter)
            
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
                if (price_below_donchian_low or 
                    not uptrend):  # Trend reversal
                    exit_signal = True
            elif position == -1:
                # Exit short: price rises above Donchian high or trend turns up
                if (price_above_donchian_high or 
                    not downtrend):  # Trend reversal
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Trend_VolatilityFilter"
timeframe = "4h"
leverage = 1.0