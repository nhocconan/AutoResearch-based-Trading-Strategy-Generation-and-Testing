#!/usr/bin/env python3
"""
Experiment #1949: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: Donchian(20) breakouts capture institutional order flow, 
filtered by 1d HMA(21) trend and volume spikes (>1.5x average) to avoid false breakouts.
Works in bull/bear markets by following HTF direction. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1949_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    half = 21 // 2
    sqrt_n = int(np.sqrt(21))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    # Compute WMA for half period
    wma_half = np.full_like(close_1d, np.nan)
    wma_half[half:] = wma(close_1d, half)
    # Compute WMA for full period
    wma_full = np.full_like(close_1d, np.nan)
    wma_full[21:] = wma(close_1d, 21)
    # HMA = WMA(2*WMA_half - WMA_full), sqrt_n
    raw_hma = 2 * wma_half - wma_full
    hma_1d = np.full_like(close_1d, np.nan)
    hma_1d[half + sqrt_n:] = wma(raw_hma[half:], sqrt_n)
    
    # 1d trend: price > HMA = bullish, price < HMA = bearish
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    donchian_len = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_len - 1, n):
        upper[i] = np.max(high[i - donchian_len + 1:i + 1])
        lower[i] = np.min(low[i - donchian_len + 1:i + 1])
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = np.full(n, np.nan)
    vol_ma[20:] = pd.Series(volume).rolling(window=20, min_periods=20).mean().values[20:]
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
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below lower Donchian band
                if price < lower[i]:
                    exit_signal = True
                # Optional: time-based exit after 10 bars (2.5 days)
                elif bars_since_entry >= 10:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above upper Donchian band
                if price > upper[i]:
                    exit_signal = True
                # Optional: time-based exit after 10 bars
                elif bars_since_entry >= 10:
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
        # Require 1d HMA trend alignment for bias filter
        trend_bias = trend_1d_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND 1d trend up
            if trend_bias > 0 and price > upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 1d trend down
            elif trend_bias < 0 and price < lower[i]:
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