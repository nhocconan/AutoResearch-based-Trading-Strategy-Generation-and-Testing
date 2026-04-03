#!/usr/bin/env python3
"""
Experiment #1970: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian channel breakouts aligned with weekly HMA trend and volume spikes capture institutional flow. 
Weekly HMA provides smooth trend filter resistant to whipsaws. Volume confirmation ensures breakout validity. 
ATR-based stoploss limits drawdown. Designed for low trade frequency (target: 30-100 total over 4 years) to minimize fee drag.
Works in both bull and bear markets by following the higher timeframe trend with precise daily entries.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1970_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate Weekly HMA(21)
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(values, period):
        if len(values) < period:
            return np.full_like(values, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    def hma(values, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        if len(values) < period:
            return np.full_like(values, np.nan)
        wma_half = wma(values, half)
        wma_full = wma(values, period)
        # Need to align arrays
        wma_half_padded = np.full_like(values, np.nan)
        wma_half_padded[half-1:len(wma_half)+half-1] = wma_half
        wma_full_padded = np.full_like(values, np.nan)
        wma_full_padded[period-1:len(wma_full)+period-1] = wma_full
        diff = 2 * wma_half_padded - wma_full_padded
        hma_values = wma(diff, sqrt_n)
        hma_padded = np.full_like(values, np.nan)
        hma_padded[sqrt_n-1:len(hma_values)+sqrt_n-1] = hma_values
        return hma_padded
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === Daily Indicators ===
    # Donchian Channel (20)
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # ATR(14) for stoploss and position sizing
    def true_range(high, low, close_prev):
        tr1 = high - low
        tr2 = np.abs(high - close_prev)
        tr3 = np.abs(low - close_prev)
        return np.maximum(tr1, np.maximum(tr2, tr3))
    
    close_prev = np.roll(close, 1)
    close_prev[0] = close[0]
    tr = true_range(high, low, close_prev)
    atr = np.zeros_like(tr)
    for i in range(len(tr)):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.mean(tr[i-13:i+1])
    
    # Volume MA(20) for spike detection
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    vol_ratio = np.ones_like(volume)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    stop_price = 0.0
    
    warmup = 50  # sufficient for all indicators
    
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
            # Stoploss: 2 * ATR below/above entry
            if position_side > 0:  # Long
                if price <= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short
                if price >= stop_price:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Additional exit: Donchian opposite touch (mean reversion)
            if position_side > 0 and price <= donch_low[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            elif position_side < 0 and price >= donch_high[i]:
                in_position = False
                position_side = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: price breaks above Donchian HIGH AND weekly trend up
            if trend_bias > 0 and price > donch_high[i]:
                in_position = True
                position_side = 1
                entry_price = price
                stop_price = entry_price - 2.0 * atr[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian LOW AND weekly trend down
            elif trend_bias < 0 and price < donch_low[i]:
                in_position = True
                position_side = -1
                entry_price = price
                stop_price = entry_price + 2.0 * atr[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals