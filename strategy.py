#!/usr/bin/env python3
"""
Experiment #249: 4h Donchian(20) Breakout + HMA(21) Trend + Volume Confirmation + ATR Stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. HMA(21) on 1d defines trend direction (bullish when price > HMA). 
Volume confirmation (>1.8x average) filters weak breakouts. ATR stoploss (2.0x) manages risk. 
Discrete position sizing (0.25) balances return and fee drag. Target: 100-180 total trades over 4 years (25-45/year).
Works in bull markets via breakout longs and in bear markets via breakout shorts (Donchian works both ways).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_249_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA(21) trend filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_2x_sub = 2 * wma_half - wma_full[len(wma_full)-len(wma_2x_sub):] if len(wma_2x_sub) > 0 else np.array([])
    hma_1d = wma(wma_2x_sub, sqrt_len) if len(wma_2x_sub) >= sqrt_len else np.array([])
    # Pad to match close_1d length
    hma_padded = np.full_like(close_1d, np.nan)
    if len(hma_1d) > 0:
        hma_padded[-len(hma_1d):] = hma_1d
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_padded)
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Trend Filter: 1d HMA(21) ---
        bullish_trend = price > hma_1d_aligned[i]
        bearish_trend = price < hma_1d_aligned[i]
        
        # --- Donchian Breakout Signals ---
        breakout_long = (price > highest_high[i-1]) and volume_spike and bullish_trend
        breakout_short = (price < lowest_low[i-1]) and volume_spike and bearish_trend
        
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
                # Exit on opposite breakout with volume
                if breakout_short:
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
                # Exit on opposite breakout with volume
                if breakout_long:
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
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals