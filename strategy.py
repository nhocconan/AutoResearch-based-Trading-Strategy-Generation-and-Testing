#!/usr/bin/env python3
"""
Experiment #305: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Spike

HYPOTHESIS: Trading Donchian channel breakouts on 12h timeframe with 1d HMA trend alignment and volume confirmation captures medium-term trends while minimizing fee drag. The 1d HMA filter ensures we only trade in the direction of the higher timeframe trend, reducing false breakouts. Volume spike confirms institutional participation. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to avoid overtrading and fee drag. Works in both bull and bear markets by trading breakouts in either direction with trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_1d_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        # 2*WMA(10.5) - WMA(21)
        diff = 2 * np.pad(wma_half, (len(close_1d) - len(wma_half), 0), mode='edge', constant_values=(np.nan, np.nan)) - np.pad(wma_full, (len(close_1d) - len(wma_full), 0), mode='edge', constant_values=(np.nan, np.nan))
        hma_1d = wma(diff[~np.isnan(diff)], sqrt_len) if len(diff[~np.isnan(diff)]) >= sqrt_len else np.full(len(close_1d), np.nan)
        # Pad to original length
        hma_1d_full = np.full(len(close_1d), np.nan)
        if len(hma_1d) > 0:
            start_idx = len(close_1d) - len(hma_1d)
            hma_1d_full[start_idx:] = hma_1d
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_full)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume Spike: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (vol_ma * 1.5)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: Only trade in direction of 1d HMA ---
        # HMA slope: rising if current > previous
        if i > 0 and not np.isnan(hma_1d_aligned[i-1]):
            hma_rising = hma_1d_aligned[i] > hma_1d_aligned[i-1]
            hma_falling = hma_1d_aligned[i] < hma_1d_aligned[i-1]
        else:
            hma_rising = False
            hma_falling = False
        
        # --- Breakout Signals ---
        # Long: Price breaks above Donchian upper band with volume spike and rising HMA
        if close[i] > donchian_high[i] and volume_spike[i] and hma_rising:
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower band with volume spike and falling HMA
        elif close[i] < donchian_low[i] and volume_spike[i] and hma_falling:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals