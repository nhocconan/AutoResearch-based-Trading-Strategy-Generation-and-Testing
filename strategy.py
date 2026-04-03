#!/usr/bin/env python3
"""
Experiment #1977: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: Donchian channel breakouts on 4h capture momentum, filtered by 1d HMA trend direction and volume spikes. 
This structure avoids whipsaws by requiring alignment with higher timeframe trend and institutional participation (volume). 
Works in both bull and bear markets by following the 1d trend direction. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1977_4h_donchian20_1d_hma_vol_v1"
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
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def calculate_hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        if half < 1 or sqrt < 1:
            return np.full_like(arr, np.nan)
        wma_half = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw = 2 * wma_half - wma_full
        hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean().values
        return hma
    
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
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
    
    warmup = 50  # sufficient for Donchian(20), HMA, volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (2*ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            tr1 = high[i] - low[i]
            tr2 = np.abs(high[i] - close[i-1])
            tr3 = np.abs(low[i] - close[i-1])
            tr = np.maximum(tr1, np.maximum(tr2, tr3))
            atr = pd.Series(tr).rolling(window=14, min_periods=1).mean().iloc[i]
            
            exit_signal = False
            
            if position_side > 0:  # Long position
                if price < entry_price - 2.0 * atr:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.0 * atr:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 1d trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d trend up
            if trend_bias > 0 and price > high_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d trend down
            elif trend_bias < 0 and price < low_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals