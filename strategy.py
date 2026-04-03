#!/usr/bin/env python3
"""
Experiment #124: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation

HYPOTHESIS: Daily Donchian channel breakouts (20-period) aligned with weekly HMA(21) trend 
and confirmed by daily volume spikes capture strong momentum moves in both bull and bear 
markets. The weekly trend filter ensures we only trade in the direction of the higher 
timeframe momentum, reducing whipsaw. Volume confirmation ensures institutional 
participation. Targets 15-25 trades/year on 1d timeframe (60-100 total over 4 years) to 
minimize fee drag while capturing high-probability breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_21 = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_padded = np.full(len(close_1w), np.nan)
        hma_padded[half_len + sqrt_len - 1:] = hma_21
        hma_21_1w = hma_padded
        
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Calculate Donchian channels (20-period) on 1d
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        donchian_high[i] = np.max(high[i - donchian_period + 1:i + 1])
        donchian_low[i] = np.min(low[i - donchian_period + 1:i + 1])
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ratio = np.full(n, np.nan)
    
    for i in range(vol_ma_period, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period:i])
        if vol_ma[i] > 0:
            vol_ratio[i] = volume[i] / vol_ma[i]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    warmup = max(100, donchian_period, vol_ma_period)  # Ensure enough data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: HMA direction ---
        hma_rising = hma_21_1w_aligned[i] > hma_21_1w_aligned[i - 1]
        hma_falling = hma_21_1w_aligned[i] < hma_21_1w_aligned[i - 1]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Breakout Conditions ---
        bullish_breakout = close[i] > donchian_high[i]
        bearish_breakout = close[i] < donchian_low[i]
        
        # --- Entry Logic ---
        # Long: Bullish breakout + rising HMA + volume spike
        if bullish_breakout and hma_rising and volume_spike:
            signals[i] = SIZE
        # Short: Bearish breakout + falling HMA + volume spike
        elif bearish_breakout and hma_falling and volume_spike:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals