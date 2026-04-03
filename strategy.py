#!/usr/bin/env python3
"""
Experiment #190: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Donchian breakouts on daily timeframe aligned with weekly HMA trend (21-period) capture high-probability continuation moves. Volume confirmation (>1.8x average) filters weak breakouts. ATR stoploss (2.5x) manages risk. Discrete position sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years. Works in both bull and bear by using breakout logic (continuation in trend direction).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_190_1d_donchian20_1w_hma_volume_v1"
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
    
    # Calculate HMA(21) on weekly close
    def calculate_hma(arr, period):
        half_len = period // 2
        sqrt_len = int(np.sqrt(period))
        
        if half_len == 0 or sqrt_len == 0:
            return arr.copy()
        
        # WMA helper
        def wma(values, window):
            weights = np.arange(1, window + 1, dtype=np.float64)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        # Pad array for edge handling
        padded = np.concatenate([np.full(period-1, np.nan), arr])
        
        wma_half = np.full_like(arr, np.nan)
        wma_full = np.full_like(arr, np.nan)
        
        for i in range(half_len-1, len(arr)+half_len-1):
            idx = i - (half_len-1)
            if idx >= 0 and idx+half_len <= len(padded):
                wma_half[idx] = wma(padded[idx:idx+half_len], half_len)[-1]
        
        for i in range(len(arr)):
            start_idx = i
            end_idx = i + period
            if end_idx <= len(padded):
                wma_full[i] = wma(padded[start_idx:end_idx], period)[-1]
        
        # HMA = 2*WMA(half) - WMA(full)
        hma_raw = 2 * wma_half - wma_full
        
        # Final WMA of sqrt_len
        hma = np.full_like(arr, np.nan)
        for i in range(sqrt_len-1, len(arr)):
            start_idx = i - (sqrt_len-1)
            end_idx = i + 1
            if hma_raw[start_idx:end_idx].size == sqrt_len:
                valid_vals = hma_raw[start_idx:end_idx]
                mask = ~np.isnan(valid_vals)
                if np.any(mask):
                    weights = np.arange(1, sqrt_len+1, dtype=np.float64)[mask]
                    vals = valid_vals[mask]
                    hma[i] = np.dot(weights, vals) / weights.sum()
        
        return hma
    
    hma_21w = calculate_hma(df_1w['close'].values, 21)
    hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21w)
    
    # === 1d Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr_1d = np.zeros(n)
    tr_1d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_1d[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_1d).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- HMA Trend Condition ---
        # HMA slope: positive = uptrend, negative = downtrend
        if i >= 1:
            hma_slope = hma_21w_aligned[i] - hma_21w_aligned[i-1]
            hma_uptrend = hma_slope > 0
            hma_downtrend = hma_slope < 0
        else:
            hma_uptrend = False
            hma_downtrend = False
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
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
            
            # Exit if breakout fails and we have minimum holding period
            if bars_since_entry >= 2:
                if position_side > 0 and not hma_uptrend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                elif position_side < 0 and not hma_downtrend:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require volume spike + breakout in trend direction
        if volume_spike:
            # Long: breakout up AND HMA uptrend
            if breakout_up and hma_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND HMA downtrend
            elif breakout_down and hma_downtrend:
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