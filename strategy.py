#!/usr/bin/env python3
"""
Experiment #369: 4h Donchian(20) + 1d Volume Spike + 1w Choppiness Regime

HYPOTHESIS: Donchian channel breakouts on 4h capture medium-term trends. Volume spikes 
on 1d confirm institutional participation. Choppiness regime on 1w filters for trending 
markets (CHOP < 38.2) to avoid whipsaws in ranging conditions. Discrete position sizing 
(0.25) minimizes fee churn. Targets 20-35 trades/year on 4h timeframe (80-140 total 
over 4 years) for optimal fee/alpha balance.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_regime_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume spike (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate volume ratio (current vs 20-period average) on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for choppiness regime (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate Choppiness Index (CHOP) on 1w
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w[1:] - low_1w[1:]
        tr2 = np.abs(high_1w[1:] - close_1w[:-1])
        tr3 = np.abs(low_1w[1:] - close_1w[:-1])
        tr = np.concatenate([[high_1w[0] - low_1w[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(sum(tr)/(hh-ll)) / log10(14)
        # Avoid division by zero
        hl_range = hh - ll
        hl_range = np.where(hl_range == 0, 1e-10, hl_range)
        chop = 100 * np.log10(tr_sum / hl_range) / np.log10(14)
        chop = np.nan_to_num(chop, nan=50.0, posinf=100.0, neginf=0.0)
        
        chop_aligned = align_htf_to_ltf(prices, df_1w, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Default to neutral if insufficient data
    
    # === LTF: 4h Donchian Channel (20) ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(lookback, 20)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when CHOP < 38.2 (trending market) ---
        is_trending = chop_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when price re-enters Donchian channel (mean reversion)
                if donchian_low[i] <= close[i] <= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit when price re-enters Donchian channel (mean reversion)
                if donchian_low[i] <= close[i] <= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian Upper with volume and trend
        long_condition = is_trending and volume_spike and (close[i] > donchian_high[i])
        
        # Short: Break below Donchian Lower with volume and trend
        short_condition = is_trending and volume_spike and (close[i] < donchian_low[i])
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals