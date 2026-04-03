#!/usr/bin/env python3
"""
Experiment #185: 12h Donchian(20) breakout + 1d HMA21 trend + volume spike + ATR stoploss
HYPOTHESIS: 12h Donchian breakouts capture medium-term momentum. 1d HMA21 provides higher timeframe trend filter to avoid counter-trend trades. Volume spike confirms institutional participation. Works in bull/bear regimes by aligning with 1d trend direction. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_185_12h_donchian_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA21 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA21 on 1d close
    close_1d = df_1d['close'].values
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    # WMA function
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1, dtype=np.float64)
        return np.convolve(arr, weights[::-1], mode='valid') / weights.sum()
    
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    wma_half = wma(close_1d, half_len)
    wma_full = wma(close_1d, 21)
    wma_half_padded = np.full_like(close_1d, np.nan)
    wma_half_padded[half_len-1:] = wma_half
    diff = 2 * wma_half_padded - wma_full
    hma_21 = wma(diff, sqrt_len)
    hma_21_padded = np.full_like(close_1d, np.nan)
    hma_21_padded[sqrt_len-1:] = hma_21
    
    trend_up_1d = close_1d > hma_21_padded
    trend_down_1d = close_1d < hma_21_padded
    
    # Align to 12h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === 12h Indicators: ATR(14) for stoploss ===
    tr_12h = np.zeros(n)
    tr_12h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_12h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Donchian Channel(20) ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(atr_14[i]) or np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i])):
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
                # Exit if price retests Donchian middle with volume
                middle = (highest_20[i] + lowest_20[i]) / 2
                if abs(price - middle) < 0.1 * atr_14[i] and volume_spike:
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
                # Exit if price retests Donchian middle with volume
                middle = (highest_20[i] + lowest_20[i]) / 2
                if abs(price - middle) < 0.1 * atr_14[i] and volume_spike:
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
        # Long: Price breaks above Donchian upper, 1d uptrend, volume spike
        if high[i] > highest_20[i] and trend_up_1d_aligned[i] and volume_spike:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower, 1d downtrend, volume spike
        elif low[i] < lowest_20[i] and trend_down_1d_aligned[i] and volume_spike:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals