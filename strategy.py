#!/usr/bin/env python3
"""
Experiment #1882: 12h Donchian(20) breakout + 1d EMA(50) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong trending moves. Combined with 1d EMA(50) trend filter and volume confirmation (>1.5x average), this strategy trades in the direction of the higher timeframe trend. Uses ATR-based stoploss (2.5*ATR) to manage risk. Target: 50-150 total trades over 4 years (12-37/year) with discrete position sizing of 0.25 to minimize fee churn. Works in both bull and bear markets by following the 1d trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1882_12h_donchian20_1d_ema_vol_v1"
timeframe = "12h"
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
    
    # === 12h Indicators: Donchian Channel (20) ===
    # Upper channel = highest high of last 20 periods
    # Lower channel = lowest low of last 20 periods
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: ATR(14) for stoploss and volatility filter ===
    # True Range
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
    bars_since_entry = 0
    highest_since_entry = 0.0  # for trailing stop (optional)
    lowest_since_entry = 0.0   # for trailing stop (optional)
    
    warmup = 50  # sufficient for EMA(50) and Donchian(20)
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(atr[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Stoploss or reverse signal ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop (optional)
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i]) if lowest_since_entry > 0 else low[i]
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5 * ATR below entry
                if price < entry_price - 2.5 * atr[i]:
                    exit_signal = True
                # Reverse signal: price breaks below lower Donchian channel
                elif price < low_roll_min[i]:
                    exit_signal = True
            else:  # Short position
                # Stoploss: 2.5 * ATR above entry
                if price > entry_price + 2.5 * atr[i]:
                    exit_signal = True
                # Reverse signal: price breaks above upper Donchian channel
                elif price > high_roll_max[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                highest_since_entry = 0.0
                lowest_since_entry = 0.0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long breakout: price closes above upper Donchian channel with bullish 1d trend
            if trend_bias > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = 0.0
                signals[i] = SIZE
            # Short breakout: price closes below lower Donchian channel with bearish 1d trend
            elif trend_bias < 0 and price < low_roll_min[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = 0.0
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals