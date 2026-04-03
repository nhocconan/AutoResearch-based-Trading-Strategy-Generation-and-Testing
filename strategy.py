#!/usr/bin/env python3
"""
Experiment #1869: 4h Donchian(20) breakout + 1d EMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. Filtered by 1d EMA trend direction and volume confirmation (>1.5x average) to avoid false breakouts. Works in both bull and bear markets by following the higher timeframe trend. Uses ATR-based stoploss to manage risk. Target: 75-200 total trades over 4 years (19-50/year) with discrete position sizing of 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1869_4h_donchian20_1d_ema_vol_v1"
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
    
    # === 4h Indicators: Donchian Channel (20) ===
    # Upper channel: highest high over last 20 periods
    # Lower channel: lowest low over last 20 periods
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    stoploss_price = 0.0
    
    warmup = 50  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reversal signal ---
        if in_position:
            bars_since_entry += 1
            
            # Check stoploss hit
            if position_side > 0:  # Long position
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if 1d trend flips against position
                elif trend_1d_aligned[i] < 0:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if 1d trend flips against position
                elif trend_1d_aligned[i] > 0:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Continue holding position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_confirm = vol_ratio[i] > 1.5
        
        if volume_confirm:
            # Long breakout: price closes above upper Donchian channel
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = price
                bars_since_entry = 0
                stoploss_price = price - 2.5 * atr[i]  # 2.5x ATR stoploss
                signals[i] = SIZE
            # Short breakout: price closes below lower Donchian channel
            elif trend_bias < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = price
                bars_since_entry = 0
                stoploss_price = price + 2.5 * atr[i]  # 2.5x ATR stoploss
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals