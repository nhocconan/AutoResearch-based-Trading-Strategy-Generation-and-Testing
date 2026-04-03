#!/usr/bin/env python3
"""
Experiment #093: 4h Donchian breakout + 12h HMA trend + volume confirmation + ATR stoploss

HYPOTHESIS: 4h Donchian(20) breakouts capture medium-term trends, filtered by 12h HMA(21) 
trend direction and 1d volume spike (>2x average) to ensure institutional participation. 
ATR(14) stoploss (2.5x) manages risk. Designed for 4h timeframe targeting 19-50 trades/year 
(75-200 total over 4 years) to minimize fee drag while working in both bull and bear markets 
via trend-following logic. Uses discrete position sizing (0.25) to reduce churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        # HMA: WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, window):
            if len(arr) < window:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_12h = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_padded = np.full(len(close_12h), np.nan)
        hma_padded[half_len:len(wma_half)+half_len] = wma_half
        hma_padded[21:len(wma_full)+21] = wma_full
        hma_padded[len(wma_diff)-sqrt_len:len(wma_diff)] = hma_12h[-sqrt_len:] if len(hma_12h) >= sqrt_len else hma_12h
        # Simpler: just use the final HMA array with proper padding
        hma_12h_final = np.full(len(close_12h), np.nan)
        if len(hma_12h) > 0:
            start_idx = len(close_12h) - len(hma_12h)
            hma_12h_final[start_idx:] = hma_12h
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_final)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Volume ratio: current vs 20-period average
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.ones(len(vol_1d))  # default 1.0
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 4h indicators: Donchian(20) and ATR(14) ===
    # Donchian channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # ATR(14)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, lookback, 21)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(atr_14[i]) or np.isnan(hma_12h_aligned[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) on 1d ---
        volume_spike = vol_ratio_1d_aligned[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long breakout: Price breaks above Donchian upper with HMA up and volume
        long_breakout = (
            close[i] > highest_high[i] and 
            close[i] > hma_12h_aligned[i] and  # Price above HMA = uptrend
            volume_spike
        )
        
        # Short breakout: Price breaks below Donchian lower with HMA down and volume
        short_breakout = (
            close[i] < lowest_low[i] and 
            close[i] < hma_12h_aligned[i] and  # Price below HMA = downtrend
            volume_spike
        )
        
        if long_breakout:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_breakout:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals