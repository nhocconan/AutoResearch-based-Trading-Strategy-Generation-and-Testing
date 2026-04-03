#!/usr/bin/env python3
"""
Experiment #028: 12h Donchian Breakout + Volume + Chop Filter Strategy

HYPOTHESIS: Uses 12h Donchian channel (20) breakouts with 1w/1d HTF trend filters, 
volume confirmation, and choppiness regime filter. Only trades in strong trends 
(ADX > 25) or low chop (CHOP < 38.2) to avoid whipsaws. Position size 0.25 to 
limit drawdown. Target: 75-150 total trades over 4 years (19-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_028_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w and 1d data for trend filters (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    
    # 1w EMA50 for long-term trend
    ema50_1w = pd.Series(df_1w['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # 1d EMA50 for medium-term trend
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # === 12h Indicators: Donchian Channel (20) ===
    def rolling_max(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.max(arr[i - window + 1:i + 1])
        return result
    
    def rolling_min(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.min(arr[i - window + 1:i + 1])
        return result
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 12h Indicators: ADX(14) for trend strength ===
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        plus_dm[i] = high[i] - high[i-1] if high[i] - high[i-1] > high[i-1] - low[i-1] and high[i] - high[i-1] > 0 else 0
        minus_dm[i] = high[i-1] - low[i-1] if high[i-1] - low[i-1] > high[i] - high[i-1] and high[i-1] - low[i-1] > 0 else 0
    
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, min_periods=14, adjust=False).mean().values / atr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 12h Indicators: Choppiness Index (14) ===
    def rolling_sum(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.sum(arr[i - window + 1:i + 1])
        return result
    
    atr_sum = rolling_sum(tr, 14)
    highest_high = rolling_max(high, 14)
    lowest_low = rolling_min(low, 14)
    chop = 100 * np.log10(atr_sum / (highest_high - lowest_low + 1e-10)) / np.log10(14)
    
    # === 12h Indicators: Volume Ratio (20) ===
    def rolling_mean(arr, window):
        result = np.full_like(arr, np.nan, dtype=np.float64)
        for i in range(window - 1, len(arr)):
            result[i] = np.mean(arr[i - window + 1:i + 1])
        return result
    
    vol_ma_20 = rolling_mean(volume, 20)
    vol_ratio = volume / (vol_ma_20 + 1e-10)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Position sizing (25% of capital)
    
    warmup = 50  # Sufficient warmup for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(adx[i]) or np.isnan(chop[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Trend Alignment: 1w and 1d EMA50 ---
        is_uptrend_htf = price > ema50_1w_aligned[i] and price > ema50_1d_aligned[i]
        is_downtrend_htf = price < ema50_1w_aligned[i] and price < ema50_1d_aligned[i]
        
        # --- Regime Filter: ADX > 25 (trending) OR Chop < 38.2 (low chop) ---
        is_trending_regime = adx[i] > 25
        is_low_chop = chop[i] < 38.2
        regime_ok = is_trending_regime or is_low_chop
        
        # --- Volume Confirmation: vol_ratio > 1.5 ---
        volume_ok = vol_ratio[i] > 1.5
        
        # --- Breakout Conditions ---
        long_breakout = price > donchian_high[i-1]  # Break above previous high
        short_breakout = price < donchian_low[i-1]  # Break below previous low
        
        # --- Entry Logic ---
        if is_uptrend_htf and long_breakout and regime_ok and volume_ok:
            signals[i] = SIZE
        elif is_downtrend_htf and short_breakout and regime_ok and volume_ok:
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals