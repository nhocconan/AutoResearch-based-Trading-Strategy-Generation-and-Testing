#!/usr/bin/env python3
"""
Experiment #163: 4h Donchian(20) Breakout + Volume Spike + 12h Chop Filter
HYPOTHESIS: 4h Donchian breakouts with volume confirmation and 12h choppiness regime filter captures institutional breakouts in both bull and bear markets. Chop filter avoids false breakouts in ranging markets. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_163_4h_donchian_vol_chop_12h_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for choppiness regime filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate Chopiness Index (14) on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr_12h = np.zeros(len(close_12h))
    tr_12h[0] = high_12h[0] - low_12h[0]
    for i in range(1, len(close_12h)):
        tr_12h[i] = max(high_12h[i] - low_12h[i], abs(high_12h[i] - close_12h[i-1]), abs(low_12h[i] - close_12h[i-1]))
    
    # ATR(14)
    atr_12h = pd.Series(tr_12h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Highest High and Lowest Low over 14 periods
    hh_12h = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll_12h = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    
    # Chopiness Index: 100 * log10(sum(ATR14) / (HH - LL)) / log10(14)
    sum_atr_14 = pd.Series(atr_12h).rolling(window=14, min_periods=14).sum().values
    chop_denom = hh_12h - ll_12h
    chop_12h = np.zeros(len(close_12h))
    mask = (chop_denom > 0) & (~np.isnan(sum_atr_14)) & (~np.isnan(chop_denom))
    chop_12h[mask] = 100 * np.log10(sum_atr_14[mask] / chop_denom[mask]) / np.log10(14)
    
    # Regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (breakout)
    chop_aligned = align_htf_to_ltf(prices, df_12h, chop_12h)
    chop_regime_trending = chop_aligned < 38.2  # Trending regime - good for breakouts
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 4h Indicators: Donchian Channel (20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(atr_14[i]) or np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(chop_regime_trending[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
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
                # Exit if price breaks below Donchian low
                if price < donch_low[i]:
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
                # Exit if price breaks above Donchian high
                if price > donch_high[i]:
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
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + volume spike + trending regime
        if price > donch_high[i] and volume_spike and chop_regime_trending[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low + volume spike + trending regime
        elif price < donch_low[i] and volume_spike and chop_regime_trending[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals