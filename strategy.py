#!/usr/bin/env python3
"""
Experiment #260: 4h Donchian(20) breakout + 1d HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture strong momentum. 1d HMA(21) defines regime: price above HMA = bull trend (long breakouts), below = bear trend (short breakouts). Volume > 1.5x average confirms conviction. ATR(14) stoploss (2.5x) limits drawdown. Discrete sizing 0.25 balances return and fee drag. Target: 75-200 total trades over 4 years (19-50/year). Works in bull via trend-following breakouts, bear via short breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_260_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA(21) for trend regime ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_21_1d = calculate_hma(close_1d, 21)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
            np.isnan(hma_21_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Trend Regime from 1d HMA ---
        bull_trend = price > hma_21_1d_aligned[i]
        bear_trend = price < hma_21_1d_aligned[i]
        
        # --- Donchian Breakout Signals ---
        breakout_long = (price > highest_20[i-1]) and volume_spike and bull_trend
        breakdown_short = (price < lowest_20[i-1]) and volume_spike and bear_trend
        
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
                # Exit on opposite breakout with volume
                if breakdown_short:
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
        elif breakdown_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

def calculate_hma(close, period):
    """Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))"""
    close = np.asarray(close)
    n = len(close)
    if n < period:
        return np.full(n, np.nan)
    
    def wma(data, window):
        weights = np.arange(1, window + 1, dtype=np.float64)
        return np.convolve(data, weights, mode='valid') / weights.sum()
    
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(close, half_period)
    wma_full = wma(close, period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    # Pad with NaN for alignment
    hma_full = np.full(n, np.nan)
    start_idx = period - 1
    end_idx = start_idx + len(hma)
    hma_full[start_idx:end_idx] = hma
    
    return hma_full