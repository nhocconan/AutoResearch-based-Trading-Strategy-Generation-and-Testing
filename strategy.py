#!/usr/bin/env python3
# 12h_Donchian20_Breakout_Volume_ATRExit
# Hypothesis: Donchian(20) breakouts on 12h timeframe capture medium-term trends in BTC/ETH/SOL.
# Volume confirmation filters false breakouts. ATR-based stop loss manages risk.
# Works in both bull and bear by capturing directional moves while avoiding chop via volume filter.
# Target: 50-150 total trades over 4 years (12-37/year) with disciplined entries.
# Uses 12h for execution, 1d for volume context and ATR calculation.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 12h data (primary) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    volume_12h = df_12h['volume'].values
    
    # === 1d data (HTF for volume context and ATR) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # === Donchian channel (20-period) on 12h ===
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donchian_high = rolling_max(high_12h, 20)
    donchian_low = rolling_min(low_12h, 20)
    
    # === ATR (14-period) on 1d for stop loss ===
    def true_range(high, low, close):
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        return np.concatenate([[np.nan], tr])
    
    tr = true_range(high_1d, low_1d, close_1d)
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Volume ratio (20-period) on 1d for confirmation ===
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ratio_1d = volume_1d / vol_ma_20_1d
    
    # Align all HTF data to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr)
    vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    
    signals = np.zeros(n)
    
    # Warmup: enough for Donchian and ATR calculations
    warmup = 50
    
    # Track position and entry price for stop loss
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(atr_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        price = close[i]
        upper = donchian_high_aligned[i]
        lower = donchian_low_aligned[i]
        atr_val = atr_aligned[i]
        vol_ratio = vol_ratio_1d_aligned[i]
        
        # === STOP LOSS LOGIC ===
        if position == 1:  # Long position
            stop_price = entry_price - 2.0 * atr_val
            if price < stop_price:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            stop_price = entry_price + 2.0 * atr_val
            if price > stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # === EXIT LOGIC (trend exhaustion) ===
        if position == 1:  # Long position
            # Exit: price closes below Donchian low (trend reversal)
            if price < lower:
                signals[i] = 0.0
                position = 0
                continue
        elif position == -1:  # Short position
            # Exit: price closes above Donchian high (trend reversal)
            if price > upper:
                signals[i] = 0.0
                position = 0
                continue
        
        # === ENTRY LOGIC (only when flat) ===
        if position == 0:
            # Require volume confirmation to avoid false breakouts
            if vol_ratio > 1.3:  # Volume above average
                # LONG: Break above Donchian high
                if price > upper:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                    continue
                # SHORT: Break below Donchian low
                elif price < lower:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
                    continue
        
        # Hold current position
        if position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
        else:
            signals[i] = 0.0
    
    return signals

name = "12h_Donchian20_Breakout_Volume_ATRExit"
timeframe = "12h"
leverage = 1.0