#!/usr/bin/env python3
"""
Experiment #1942: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture significant trends, 
filtered by 1d HMA(21) trend direction and volume spikes (>1.5x 20-period average).
This structure avoids overtrading while maintaining edge in both bull and bear markets.
Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1942_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
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
    
    # Calculate 1d HMA(21)
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        hma_raw = 2 * wma_half - wma_full
        hma = wma(hma_raw, sqrt_period)
        
        # Pad with NaN to match original length
        result = np.full_like(arr, np.nan, dtype=np.float64)
        start_idx = period - 1
        end_idx = start_idx + len(hma)
        result[start_idx:end_idx] = hma
        return result
    
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_21_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    def donchian_channels(high_arr, low_arr, period):
        upper = pd.Series(high_arr).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low_arr).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channels(high, low, 20)
    
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
    
    warmup = 50  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: opposite Donchian touch or trend reversal
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price touches or breaks below lower Donchian
                if price <= donchian_lower[i]:
                    exit_signal = True
                # Exit if 1d trend turns bearish
                elif trend_1d_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price touches or breaks above upper Donchian
                if price >= donchian_upper[i]:
                    exit_signal = True
                # Exit if 1d trend turns bullish
                elif trend_1d_aligned[i] > 0:
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
            # Long entry: price breaks above upper Donchian AND 1d trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND 1d trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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