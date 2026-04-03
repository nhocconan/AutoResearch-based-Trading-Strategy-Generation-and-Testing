#!/usr/bin/env python3
"""
Experiment #324: 1d Donchian Breakout + 1w Volume Confirmation + Regime Filter

HYPOTHESIS: Daily Donchian(20) breakouts confirmed by weekly volume spikes and 
choppiness regime filter capture sustainable momentum moves. Weekly timeframe 
provides structural context while daily entries minimize lag. Discrete position 
sizing (0.25) and ATR-based stops control risk. Target: 50-100 total trades 
over 4 years (12-25/year) to minimize fee drag in bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_vol_regime_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for volume confirmation (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 20:
        vol_1w = df_1w['volume'].values
        vol_ma_20 = pd.Series(vol_1w).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1w = np.zeros(len(vol_1w))
        vol_ratio_1w[20:] = vol_1w[20:] / vol_ma_20[20:]
        vol_ratio_1w[:20] = 1.0
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, 1.0)
    
    # === HTF: 1w data for choppiness regime (Call ONCE before loop) ===
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = high_1w[0] - low_1w[0]  # First bar
        
        # ATR(14)
        atr_1w = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Highest High and Lowest Low over 14 periods
        hh_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
        ll_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
        
        # Chopiness Index: 100 * log10(sum(atr/14) / log(hh/ll)) / log10(14)
        sum_atr = pd.Series(atr_1w).rolling(window=14, min_periods=14).sum().values
        ratio = np.where((hh_1w > ll_1w) & (ll_1w > 0), sum_atr / (hh_1w - ll_1w), 1.0)
        chop_1w = 100 * np.log10(ratio) / np.log10(14)
        chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    else:
        chop_1w_aligned = np.full(n, 50.0)  # Neutral chop
    
    # === 1d Indicators: Donchian Channel (20) ===
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
        # Warmup period: use expanding window
        for i in range(20):
            donchian_high[i] = np.max(high[:i+1])
            donchian_low[i] = np.min(low[:i+1])
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ratio_1w_aligned[i]) or np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade in trending markets (Chop < 38.2) ---
        is_trending = chop_1w_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_1w_aligned[i] > 2.0
        
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
                # Take profit at Donchian Low (trailing stop)
                if close[i] <= donchian_low[i]:
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
                # Take profit at Donchian High (trailing stop)
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Break above Donchian High with volume confirmation in trending market
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            is_trending
        )
        
        # Short: Break below Donchian Low with volume confirmation in trending market
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            is_trending
        )
        
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