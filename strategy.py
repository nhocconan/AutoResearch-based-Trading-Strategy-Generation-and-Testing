#!/usr/bin/env python3
"""
Experiment #2859: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: 6h Donchian(20) breakouts aligned with weekly pivot direction (from 1w Camarilla R4/S4) 
and volume confirmation capture institutional breakout moves. Weekly pivot direction ensures we 
only trade breakouts in the direction of the weekly trend (buy breakouts in weekly uptrend, 
sell breakdowns in weekly downtrend), reducing false breakouts in ranging markets. Volume 
confirmation adds conviction. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_2859_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels for trend direction
    # Pivot = (High + Low + Close) / 3
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    range_1w = high_1w - low_1w
    
    # Weekly Camarilla R4/S4 levels (strong breakout levels)
    r4_1w = pivot_1w + range_1w * 1.1
    s4_1w = pivot_1w - range_1w * 1.1
    
    # Weekly trend bias: 1 = bullish (price above R4), -1 = bearish (price below S4), 0 = neutral
    trend_bias_1w = np.where(close_1w > r4_1w, 1, np.where(close_1w < s4_1w, -1, 0))
    trend_bias_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_bias_1w)
    
    # === 6h Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
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
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_bias_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2*ATR below highest since entry
                atr_estimate = (high[i] - low[i]) * 0.5  # rough ATR estimate
                if price < highest_since_entry - 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Donchian level (take profit)
                elif position_side > 0 and price >= lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2*ATR above lowest since entry
                atr_estimate = (high[i] - low[i]) * 0.5
                if price > lowest_since_entry + 2.0 * atr_estimate:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price reaches opposite Donchian level (take profit)
                elif position_side < 0 and price <= highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.8x average) for confirmation
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Get weekly trend bias
            weekly_bias = trend_bias_1w_aligned[i]
            
            # Long entry: price breaks above 6h Donchian(20) high in weekly bullish bias
            # (breakout continuation in bullish weekly trend)
            if weekly_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 6h Donchian(20) low in weekly bearish bias
            # (breakdown continuation in bearish weekly trend)
            elif weekly_bias < 0 and price < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals