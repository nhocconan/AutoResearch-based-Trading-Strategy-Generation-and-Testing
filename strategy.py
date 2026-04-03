#!/usr/bin/env python3
"""
Experiment #106: 4h Donchian(20) + 1d Volume Confirmation + Chop Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe with 1d volume spike confirmation 
and choppiness regime filter (CHOP > 61.8 for mean reversion, CHOP < 38.2 for trend) 
creates a robust strategy that works in both bull and bear markets. The Donchian 
structure provides objective breakout levels, volume confirms institutional 
participation, and the chop filter adapts to market conditions. Targets 19-50 
trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag while 
capturing high-probability breakouts and mean reversion swings.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_1d_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
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
    
    # === HTF: 1d data for chop regime filter (Call ONCE before loop) ===
    if len(df_1d) >= 14:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # True Range
        tr = np.zeros(len(close_1d))
        tr[0] = high_1d[0] - low_1d[0]
        for i in range(1, len(close_1d)):
            tr[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]), 
                       abs(low_1d[i] - close_1d[i-1]))
        
        # ATR(14)
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Sum of True Range over 14 periods
        tr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
        
        # Max High - Min Low over 14 periods
        max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
        range_maxmin = max_high - min_low
        
        # Choppiness Index: CHOP = 100 * log10(TR_sum / (ATR * 14)) / log10(14)
        chop = np.zeros(len(close_1d))
        mask = (atr_14 > 0) & (range_maxmin > 0) & (tr_sum > 0)
        chop[mask] = 100 * np.log10(tr_sum[mask] / (atr_14[mask] * 14)) / np.log10(14)
        chop[~mask] = 50.0  # Neutral when invalid
        
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)
    
    # === 4h Indicators ===
    # Donchian(20) channels
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    
    for i in range(n):
        if i >= 19:
            donchian_high[i] = np.max(high[i-19:i+1])
            donchian_low[i] = np.min(low[i-19:i+1])
        else:
            donchian_high[i] = np.nan
            donchian_low[i] = np.nan
    
    # ATR(14) for stoploss on 4h
    tr_4h = np.zeros(n)
    tr_4h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_4h[i] = max(high[i] - low[i], 
                      abs(high[i] - close[i-1]), 
                      abs(low[i] - close[i-1]))
    atr_14_4h = pd.Series(tr_4h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
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
            np.isnan(vol_ratio_1d_aligned[i]) or np.isnan(chop_aligned[i]) or
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: CHOP < 38.2 = trending (breakout), CHOP > 61.8 = ranging (mean reversion) ---
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14_4h[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band
                if close[i] >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14_4h[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at opposite Donchian band
                if close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long Logic
        long_condition = False
        if is_trending:
            # Trending market: breakout above Donchian high with volume
            long_condition = (close[i] > donchian_high[i]) and volume_spike
        elif is_ranging:
            # Ranging market: mean reversion from Donchian low
            long_condition = (close[i] <= donchian_low[i] * 1.001) and volume_spike
        
        # Short Logic
        short_condition = False
        if is_trending:
            # Trending market: breakdown below Donchian low with volume
            short_condition = (close[i] < donchian_low[i]) and volume_spike
        elif is_ranging:
            # Ranging market: mean reversion from Donchian high
            short_condition = (close[i] >= donchian_high[i] * 0.999) and volume_spike
        
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