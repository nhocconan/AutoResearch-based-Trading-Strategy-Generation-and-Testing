#!/usr/bin/env python3
"""
Experiment #1923: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: Donchian(20) breakouts on 4h capture medium-term trends. 12h HMA(21) provides trend filter to avoid counter-trend trades. Volume confirmation (>1.5x 20-period average) ensures institutional participation. Works in bull/bear markets by following the 12h trend direction. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1923_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        raw = 2 * wma_half - wma_full
        return wma(raw, sqrt_n)
    
    # Pad WMA results to match original length
    hma_values = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 21:
        wma_half = wma(close_12h, 10)
        wma_full = wma(close_12h, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw = 2 * wma_half - wma_full
            wma_raw = wma(raw, int(np.sqrt(21)))
            start_idx = 21 - 1  # Account for WMA padding
            end_idx = start_idx + len(wma_raw)
            if end_idx <= len(close_12h):
                hma_values[start_idx:end_idx] = wma_raw
    
    trend_12h = np.where(close_12h > hma_values, 1, -1)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # === 4h Indicators: Donchian(20) and Volume MA(20) ===
    # Donchian(20): upper = max(high,20), lower = min(low,20)
    high_roll_max = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_min = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    warmup = 20  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll_max[i]) or np.isnan(low_roll_min[i]) or
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: opposite Donchian breakout or trend reversal
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian lower (20) OR trend turns bearish
                if price < low_roll_min[i] or trend_12h_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian upper (20) OR trend turns bullish
                if price > high_roll_max[i] or trend_12h_aligned[i] > 0:
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper (20) AND 12h trend up
            if trend_12h_aligned[i] > 0 and price > high_roll_max[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower (20) AND 12h trend down
            elif trend_12h_aligned[i] < 0 and price < low_roll_min[i]:
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