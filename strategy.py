#!/usr/bin/env python3
"""
Experiment #1928: 12h Donchian(20) breakout + 1w HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts capture intermediate-term trends while 1w HMA filter ensures alignment with weekly momentum. 
Volume confirmation adds conviction, ATR stoploss manages risk. Works in bull/bear by following 1w trend direction.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1928_12h_donchian20_1w_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        # Pad to original length
        wma_half_pad = np.full_like(arr, np.nan)
        wma_full_pad = np.full_like(arr, np.nan)
        wma_half_pad[half-1:] = wma_half
        wma_full_pad[period-1:] = wma_full
        diff = 2 * wma_half_pad - wma_full_pad
        return wma(diff, sqrt_n)
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 12h Indicators: Donchian(20) channels ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donch_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donch_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
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
    bars_since_entry = 0
    highest_since_entry = 0.0  # for trailing stop (long)
    lowest_since_entry = 0.0   # for trailing stop (short)
    
    warmup = max(lookback, 20, 14)  # sufficient for Donchian, volume MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Time-based exit: max 10 bars (5 days for 12h)
            if bars_since_entry >= 10:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1w trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian Upper AND 1w trend up
            if trend_bias > 0 and close[i] > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian Lower AND 1w trend down
            elif trend_bias < 0 and close[i] < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals