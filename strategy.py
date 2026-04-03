#!/usr/bin/env python3
"""
Experiment #1860: 4h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Filtered by 1d EMA(50) trend direction and volume spike (>1.5x average) to avoid false breakouts. ATR-based stoploss (2.5x ATR) manages risk. Works in both bull and bear markets by following the higher timeframe trend. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1860_4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_1d = np.where(close_1d > ema_50_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian Channel(20) ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    dc_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    dc_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 4h Indicators: ATR(14) for stoploss and volatility filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    bars_since_entry = 0
    
    warmup = max(50, 20, 20, 14)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss check
            if position_side > 0:  # Long position
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Reverse signal check (opposite breakout with trend and volume)
            if position_side > 0:  # Long - check for short signal
                if (price < dc_lower[i] and 
                    trend_1d_aligned[i] < 0 and 
                    vol_ratio[i] > 1.5):
                    in_position = False
                    position_side = -1
                    entry_price = price
                    stop_price = price + 2.5 * atr[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = SIZE
            else:  # Short - check for long signal
                if (price > dc_upper[i] and 
                    trend_1d_aligned[i] > 0 and 
                    vol_ratio[i] > 1.5):
                    in_position = False
                    position_side = 1
                    entry_price = price
                    stop_price = price - 2.5 * atr[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper with bullish 1d trend
            if (price > dc_upper[i] and 
                trend_bias > 0):
                in_position = True
                position_side = 1
                entry_price = price
                stop_price = price - 2.5 * atr[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower with bearish 1d trend
            elif (price < dc_lower[i] and 
                  trend_bias < 0):
                in_position = True
                position_side = -1
                entry_price = price
                stop_price = price + 2.5 * atr[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals