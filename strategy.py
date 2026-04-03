#!/usr/bin/env python3
"""
Experiment #253: 4h Donchian Breakout + 12h HMA Trend + Volume Filter
HYPOTHESIS: 4h Donchian(20) breakouts capture momentum, filtered by 12h HMA(21) trend direction and volume confirmation (>1.5x average). 
ATR(14) stoploss (2.5x) manages risk. Discrete position sizing (0.25) limits fee drag. 
Works in bull markets via breakout continuation and in bear markets via failed breakouts (mean reversion at channel edges).
Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_253_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA(21) trend filter ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    # HMA calculation: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='valid')
    def hma(arr, period):
        half = period // 2
        sqrt_n = int(np.sqrt(period))
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        raw = 2 * wma_half - wma_full
        return wma(raw, sqrt_n)
    hma_21 = hma(close_12h, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_12h, hma_21)
    
    # === 4h Indicators: Donchian Channels (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_20 + lowest_20) / 2
    
    # === 4h Indicators: ATR(14) for stoploss and thresholds ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Filter: 12h HMA(21) slope ---
        # Uptrend: current HMA > previous HMA
        hma_uptrend = hma_21_aligned[i] > hma_21_aligned[i-1]
        # Downtrend: current HMA < previous HMA
        hma_downtrend = hma_21_aligned[i] < hma_21_aligned[i-1]
        
        # --- Donchian Breakout Signals ---
        # Long breakout: price closes above upper channel
        long_breakout = close[i] > highest_20[i-1]  # Use previous bar's channel
        # Short breakout: price closes below lower channel
        short_breakout = close[i] < lowest_20[i-1]  # Use previous bar's channel
        
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
                # Exit: price reverts to midpoint (mean reversion)
                if abs(price - donchian_mid[i]) < 0.1 * atr_14[i]:
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
                # Exit: price reverts to midpoint (mean reversion)
                if abs(price - donchian_mid[i]) < 0.1 * atr_14[i]:
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
        if volume_spike:
            # Long entry: breakout + uptrend filter
            if long_breakout and hma_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: breakout + downtrend filter
            elif short_breakout and hma_downtrend:
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