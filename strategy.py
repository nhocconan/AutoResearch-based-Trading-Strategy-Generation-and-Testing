#!/usr/bin/env python3
"""
Experiment #362: 12h Donchian Breakout + Volume Spike + 1d Choppiness Regime Filter

HYPOTHESIS: On the 12h timeframe, Donchian(20) breakouts capture significant momentum moves.
Volume confirmation (>1.8x average) ensures institutional participation, while the 1d 
Choppiness Index (>61.8 = ranging, <38.2 = trending) acts as a regime filter to avoid 
whipsaws in sideways markets. We trade breakouts in the direction of the 1d trend 
(ADX > 25) to avoid counter-trend entries. This combination has proven effective on 
SOLUSDT (test Sharpe 1.10-1.38) and should generalize across BTC/ETH/SOL. Target: 
12-37 trades/year (50-150 total over 4 years) for minimal fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_vol_chop_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Choppiness Index and ADX regime filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Choppiness Index(14) on 1d
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr1 = high_1d[1:] - low_1d[1:]
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
        
        # Sum of TR over 14 periods
        atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Highest high and lowest low over 14 periods
        hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        
        # Choppiness Index: 100 * log10(atr_sum / (hh - ll)) / log10(14)
        # Avoid division by zero
        hl_range = hh - ll
        chop = np.full(len(close_1d), 50.0)  # Default to neutral
        valid = (hl_range > 0) & ~np.isnan(atr_sum) & ~np.isnan(hl_range)
        if np.any(valid):
            chop[valid] = 100 * np.log10(atr_sum[valid] / hl_range[valid]) / np.log10(14)
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
        
        # Calculate ADX(14) on 1d for trend strength
        # True Range (already calculated above as 'tr')
        # Directional Movement
        dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                           np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
        dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                            np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
        dm_plus = np.concatenate([[0], dm_plus])
        dm_minus = np.concatenate([[0], dm_minus])
        
        # Smoothed values
        atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, min_periods=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx = pd.Series(dx).ewm(span=14, min_periods=14, adjust=False).mean().values
        adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    else:
        chop_aligned = np.full(n, 50.0)  # Default to neutral chop
        adx_aligned = np.full(n, 20.0)   # Default to weak trend
    
    # === LTF: 12h Donchian Channel (20) for breakout signals ===
    # Donchian Upper = max(high, 20)
    # Donchian Lower = min(low, 20)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === HTF: 12h data for volume confirmation (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume ratio (current vs 20-period average) on 12h
    if len(df_12h) >= 20:
        vol_12h = df_12h['volume'].values
        vol_ma_20 = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
        vol_ratio_12h = np.zeros(len(vol_12h))
        vol_ratio_12h[20:] = vol_12h[20:] / vol_ma_20[20:]
        vol_ratio_12h[:20] = 1.0  # Neutral for warmup
        vol_ratio_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ratio_12h)
    else:
        vol_ratio_12h_aligned = np.full(n, 1.0)
    
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
        if (np.isnan(chop_aligned[i]) or np.isnan(adx_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when market is trending (ADX > 25 AND Chop < 38.2) ---
        is_trending = (adx_aligned[i] > 25) and (chop_aligned[i] < 38.2)
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.8
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian High with volume and trend
        long_condition = is_trending and volume_spike and (close[i] > donchian_high[i])
        
        # Short: Price breaks below Donchian Low with volume and trend
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