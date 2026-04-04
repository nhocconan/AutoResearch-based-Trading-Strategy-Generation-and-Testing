#!/usr/bin/env python3
"""
Experiment #4338: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian breakouts capture medium-term trends. Weekly HMA(21) filters direction to avoid counter-trend trades. Volume >1.5x average confirms breakout strength. Works in bull via upside breakouts with rising volume, in bear via downside breakdowns. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4338_1d_donchian20_1w_hma_vol_v1"
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
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        wma_full = np.concatenate([np.full(20, np.nan), wma(close_1w, 21)])
        wma_half = np.concatenate([np.full(half_len-1, np.nan), wma(close_1w, half_len)])
        raw_hma = 2 * wma_half - wma_full
        hma_1w = np.concatenate([np.full(sqrt_len-1, np.nan), wma(raw_hma[~np.isnan(raw_hma)], sqrt_len)])
        # Pad to original length
        hma_1w_full = np.full(len(close_1w), np.nan)
        hma_1w_full[20:] = hma_1w[:len(close_1w)-20]
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    warmup = max(20, 20)  # Donchian, vol MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(hma_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Entry Logic ---
        # Long: price breaks above Donchian high + volume confirmation + price > weekly HMA (uptrend)
        long_breakout = price > donch_high[i-1]  # Break above previous period's high
        long_vol = vol_ratio[i] > 1.5
        long_trend = price > hma_1w_aligned[i]
        
        # Short: price breaks below Donchian low + volume confirmation + price < weekly HMA (downtrend)
        short_breakout = price < donch_low[i-1]  # Break below previous period's low
        short_vol = vol_ratio[i] > 1.5
        short_trend = price < hma_1w_aligned[i]
        
        if long_breakout and long_vol and long_trend:
            signals[i] = SIZE
        elif short_breakout and short_vol and short_trend:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals