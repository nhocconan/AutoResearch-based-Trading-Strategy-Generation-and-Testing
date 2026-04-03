#!/usr/bin/env python3
"""
Experiment #1960: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts capture institutional entry/exit points. 
Combined with 1d HMA trend filter and volume confirmation, this strategy 
targets high-probability breakouts in both bull and bear markets. 
Breakouts above Donchian(20) high with 1d HMA uptrend and volume spike = long.
Breakouts below Donchian(20) low with 1d HMA downtrend and volume spike = short.
Exit via ATR trailing stop or opposite Donchian touch.
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1960_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21): Weighted Moving Average of WMA
    # HMA = WMA(2 * WMA(n/2) - WMA(n)), sqrt(n)
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(arr, period):
        if period <= 0:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='same')
    
    # Handle NaN at edges
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_1d = wma(raw_hma, sqrt_len)
    
    # Trend: 1 = uptrend (close > HMA), -1 = downtrend (close < HMA)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    lookback = 20
    # Calculate rolling max/min
    high_roll_max = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    low_roll_min = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donch_high = high_roll_max
    donch_low = low_roll_min
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0  # first bar has no previous close
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0  # for trailing stop (long)
    lowest_since_entry = 0.0   # for trailing stop (short)
    bars_since_entry = 0
    
    warmup = max(lookback, 20, 14) + 5  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price touches opposite Donchian low (mean reversion)
                if price <= donch_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # ATR trailing stop: exit if price drops 2.5*ATR from high
                elif price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                    
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price touches opposite Donchian high
                if price >= donch_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # ATR trailing stop: exit if price rises 2.5*ATR from low
                elif price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # If no exit, maintain position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d HMA trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian high AND 1d HMA uptrend
            if trend_bias > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian low AND 1d HMA downtrend
            elif trend_bias < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals