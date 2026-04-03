#!/usr/bin/env python3
"""
Experiment #169: 4h Donchian(20) Breakout + 1d/1w HMA Trend + Volume Spike
HYPOTHESIS: 4h Donchian breakouts aligned with 1d/1w HMA trend direction and volume confirmation capture institutional breakout moves. Works in bull/bear regimes by using HTF trend filter. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_169_4h_donchian_1d1w_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA21 trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    trend_up_1d = close_1d > hma_21_1d
    trend_down_1d = close_1d < hma_21_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === HTF: 1w data for HMA21 trend filter (stronger regime) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    hma_21_1w = calculate_hma(close_1w, 21)
    trend_up_1w = close_1w > hma_21_1w
    trend_down_1w = close_1w < hma_21_1w
    trend_up_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_up_1w)
    trend_down_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_down_1w)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
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
        if (np.isnan(atr_14[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(trend_up_1w_aligned[i]) or np.isnan(trend_down_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
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
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Require BOTH 1d AND 1w trend alignment for stronger filter
        strong_uptrend = trend_up_1d_aligned[i] and trend_up_1w_aligned[i]
        strong_downtrend = trend_down_1d_aligned[i] and trend_down_1w_aligned[i]
        
        # Long: Price breaks above Donchian(20) high, strong uptrend on both TFs, volume spike
        if price > highest_20[i] and strong_uptrend and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian(20) low, strong downtrend on both TFs, volume spike
        elif price < lowest_20[i] and strong_downtrend and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(arr, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    half = period // 2
    sqrt = int(np.sqrt(period))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan, dtype=np.float64)
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(values, weights, mode='valid') / weights.sum()
    
    wma_half = wma(arr, half)
    wma_full = wma(arr, period)
    
    if len(wma_half) == 0 or len(wma_full) == 0:
        return np.full_like(arr, np.nan, dtype=np.float64)
    
    raw = 2 * wma_half[-len(wma_full):] - wma_full
    hma = wma(raw, sqrt)
    
    # Pad with NaN to match original length
    result = np.full_like(arr, np.nan, dtype=np.float64)
    start_idx = period - len(hma)
    if start_idx >= 0 and start_idx < len(arr):
        result[start_idx:] = hma
    return result