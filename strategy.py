#!/usr/bin/env python3
"""
Experiment #183: 4h Donchian(20) + HMA(21) Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Combines price channel breakout (Donchian) with Hull Moving Average trend filter and volume confirmation to capture institutional breakouts in both bull and bear markets. Uses 12h HTF for regime alignment. Target: 75-200 total trades over 4 years with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_183_4h_donchian_hma_vol_12h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close for trend direction
    def calculate_hma(arr, period):
        half_len = period // 2
        sqrt_len = int(np.sqrt(period))
        wma_half = pd.Series(arr).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        hma_raw = 2 * wma_half - wma_full
        hma = pd.Series(hma_raw).ewm(span=sqrt_len, adjust=False).mean().values
        return hma
    
    close_12h = df_12h['close'].values
    hma_21_12h = calculate_hma(close_12h, 21)
    hma_uptrend_12h = close_12h > hma_21_12h
    hma_downtrend_12h = close_12h < hma_21_12h
    
    # Align 12h HMA trend to 4h timeframe
    hma_uptrend_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_uptrend_12h)
    hma_downtrend_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_downtrend_12h)
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian Channels (20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    hma_21_4h = calculate_hma(close, 21)
    hma_uptrend_4h = close > hma_21_4h
    hma_downtrend_4h = close < hma_21_4h
    
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
        if (np.isnan(atr_14[i]) or np.isnan(hma_21_4h[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(hma_uptrend_12h_aligned[i]) or np.isnan(hma_downtrend_12h_aligned[i])):
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
                # Exit if price breaks below Donchian low with volume
                if price < lowest_20[i] and volume_spike:
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
                # Exit if price breaks above Donchian high with volume
                if price > highest_20[i] and volume_spike:
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
        # Long: Price breaks above Donchian high, 12h uptrend, 4h HMA uptrend, volume spike
        if (price > highest_20[i] and 
            hma_uptrend_12h_aligned[i] and 
            hma_uptrend_4h[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low, 12h downtrend, 4h HMA downtrend, volume spike
        elif (price < lowest_20[i] and 
              hma_downtrend_12h_aligned[i] and 
              hma_downtrend_4h[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals