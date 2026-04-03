#!/usr/bin/env python3
"""
Experiment #185: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts aligned with 1d HMA(21) trend capture sustained moves with low whipsaw. Volume (>1.5x avg) confirms breakout strength. ATR(14) stoploss (2.0x) manages risk. Discrete sizing (0.25) minimizes fee churn. Target: 75-150 total trades over 4 years. Works in bull/bear via trend filter and breakout symmetry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_185_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) for 1d
    def calculate_hma(arr, period):
        """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
        half_period = max(1, period // 2)
        sqrt_period = max(1, int(np.sqrt(period)))
        
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1, dtype=np.float64)
            weights_sum = weights.sum()
            result = np.full_like(values, np.nan)
            for i in range(window - 1, len(values)):
                result[i] = np.dot(values[i - window + 1:i + 1], weights) / weights_sum
            return result
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt_period)
        return hma
    
    hma_1d = calculate_hma(df_1d['close'].values, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Filter: price > HMA for long, price < HMA for short ---
        hma_trend_up = price > hma_1d_aligned[i]
        hma_trend_down = price < hma_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
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
        # Require volume spike + breakout conditions aligned with HTF trend
        if volume_spike:
            # Long: breakout up AND HTF trend up
            if breakout_up and hma_trend_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout down AND HTF trend down
            elif breakout_down and hma_trend_down:
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