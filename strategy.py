#!/usr/bin/env python3
"""
Experiment #1945: 12h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts capture medium-term momentum while 1d HMA filter ensures alignment with higher timeframe trend. Volume confirmation adds validity. Works in both bull/bear markets by following institutional flow on 12h timeframe with proper risk control. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1945_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend and ATR (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21)
    def calculate_hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1:
            half_period = 1
        if sqrt_period < 1:
            sqrt_period = 1
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights, 'valid') / weights.sum()
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt_period)
        # Pad to original length
        hma_padded = np.full(len(arr), np.nan, dtype=np.float64)
        start_idx = len(arr) - len(hma)
        hma_padded[start_idx:] = hma
        return hma_padded
    
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Calculate 1d ATR(14) for stoploss
    def calculate_atr(high, low, close, period):
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        atr = np.zeros_like(tr)
        for i in range(len(tr)):
            if i < period:
                atr[i] = np.mean(tr[:i+1])
            else:
                atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        return atr
    
    atr_14_1d = calculate_atr(high_1d, low_1d, close_1d, 14)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # === 12h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian channels
    high_ma = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_ma = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_ma
    donchian_lower = low_ma
    
    # Volume MA(20) for spike detection
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
    stoploss_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Stoploss: 2 * ATR below/above entry
            if position_side > 0:  # Long
                if price <= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price >= stoploss_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Exit on Donchian opposite touch (trailing stop)
            if position_side > 0 and price <= donchian_lower[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price >= donchian_upper[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d trend up
            if trend_1d_aligned[i] > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                stoploss_price = entry_price - 2.0 * atr_14_1d_aligned[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d trend down
            elif trend_1d_aligned[i] < 0 and price < donchian_lower[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                stoploss_price = entry_price + 2.0 * atr_14_1d_aligned[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals