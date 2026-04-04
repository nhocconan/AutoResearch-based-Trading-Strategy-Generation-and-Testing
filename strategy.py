#!/usr/bin/env python3
"""
Experiment #4344: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum moves. 1w HMA(21) ensures alignment with higher timeframe trend. Volume >1.5x average confirms institutional participation. ATR-based trailing stop manages risk. Works in bull via upside breakouts, in bear via downside breakouts. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4344_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1w HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21): WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1, dtype=np.float64)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values.astype(np.float64)
        wma_half = wma(close_1w, half)
        wma_full = wma(close_1w, 21)
        # Align arrays: wma_half starts at index (21-half), wma_full at index 0
        raw_hma = 2 * wma_half - wma_full[21-half:] if len(wma_half) > 0 else np.array([])
        hma_1w = wma(raw_hma, sqrt_n) if len(raw_hma) >= sqrt_n else np.array([])
        # Pad to match df_1w length
        hma_1w_full = np.full(len(df_1w), np.nan)
        start_idx = 21 - half + sqrt_n - 1
        if start_idx < len(hma_1w_full) and len(hma_1w) > 0:
            end_idx = start_idx + len(hma_1w)
            if end_idx <= len(hma_1w_full):
                hma_1w_full[start_idx:end_idx] = hma_1w
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter: price > 1w HMA for long, price < 1w HMA for short
        hma_trend_long = price > hma_1w_aligned[i]
        hma_trend_short = price < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = price > highest_high[i]
        short_breakout = price < lowest_low[i]
        
        if volume_confirm:
            # Long conditions: Donchian upside breakout + price > 1w HMA
            if long_breakout and hma_trend_long:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short conditions: Donchian downside breakout + price < 1w HMA
            elif short_breakout and hma_trend_short:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

</think>