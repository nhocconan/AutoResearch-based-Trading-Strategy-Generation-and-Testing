#!/usr/bin/env python3
"""
Experiment #030: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike

HYPOTHESIS: Combines Donchian channel breakouts on 1d with 1week HMA trend filter and volume confirmation.
In strong weekly trends (price above/below HMA(21)), we take breakouts of the daily Donchian(20) channel.
Volume must be > 1.5x 20-day average to confirm breakout strength. Uses discrete position sizing (0.25) 
and ATR-based stoploss to manage risk. Designed to work in both bull (breakouts continuation) and 
bear (breakdown continuation) markets by following the weekly trend direction. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_030_1d_donchian20_1w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate 1w HMA(21) for trend filter
    def hull_moving_average(arr, period):
        """Hull Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan, dtype=np.float64)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA(period/2)
        wma_half = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(half_period - 1, len(arr)):
            wma_half[i] = np.sum(arr[i - half_period + 1:i + 1] * np.arange(1, half_period + 1)) / (half_period * (half_period + 1) / 2)
        
        # WMA(period)
        wma_full = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(period - 1, len(arr)):
            wma_full[i] = np.sum(arr[i - period + 1:i + 1] * np.arange(1, period + 1)) / (period * (period + 1) / 2)
        
        # HMA = 2*WMA(period/2) - WMA(period)
        hma_raw = 2 * wma_half - wma_full
        
        # WMA(sqrt(period)) of the above
        hma = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(sqrt_period - 1, len(arr)):
            hma[i] = np.sum(hma_raw[i - sqrt_period + 1:i + 1] * np.arange(1, sqrt_period + 1)) / (sqrt_period * (sqrt_period + 1) / 2)
        
        return hma
    
    hma_21w = hull_moving_average(df_1w['close'].values, 21)
    hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21w)
    
    # === 1d Indicators: Donchian(20) ===
    def donchian_channels(high, low, period):
        """Donchian Channels: upper=max(high,period), lower=min(low,period)"""
        upper = np.full_like(high, np.nan, dtype=np.float64)
        lower = np.full_like(low, np.nan, dtype=np.float64)
        for i in range(period - 1, len(high)):
            upper[i] = np.max(high[i - period + 1:i + 1])
            lower[i] = np.min(low[i - period + 1:i + 1])
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume Spike (volume > 1.5x 20-day average) ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Sufficient warmup for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21w_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(atr_14[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Weekly Trend Filter ---
        is_uptrend_1w = price > hma_21w_aligned[i]
        is_downtrend_1w = price < hma_21w_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            # ATR-based stoploss
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if weekly trend changes
                if not is_uptrend_1w:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if weekly trend changes
                if not is_downtrend_1w:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Minimum holding period of 3 days to reduce churn
            if bars_since_entry < 3:
                signals[i] = position_side * SIZE
                continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Weekly uptrend + Donchian breakout above upper + volume spike
        if is_uptrend_1w and price > upper_20[i] and volume_spike[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Weekly downtrend + Donchian breakdown below lower + volume spike
        elif is_downtrend_1w and price < lower_20[i] and volume_spike[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
    
    return signals