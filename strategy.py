#!/usr/bin/env python3
"""
Experiment #393: 4h Donchian Breakout + 12h Volume Spike + 4h Chop Regime Filter

HYPOTHESIS: Donchian(20) breakout on 4h timeframe with 12h volume confirmation and 4h chop regime filter (CHOP > 61.8 = range, < 38.2 = trend) creates a robust strategy. In trending regimes (CHOP < 38.2), we trade breakouts; in ranging regimes (CHOP > 61.8), we fade Donchian touches. Volume spike confirms institutional participation. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to minimize fee drag while capturing high-probability breakouts and mean reversions at key levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_vol_chop_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for volume spike (Call ONCE before loop) ===
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
    
    # === 4h Indicators ===
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20, n):
            donchian_high[i] = np.max(high[i-20:i])
            donchian_low[i] = np.min(low[i-20:i])
    
    # Choppiness Index (CHOP) on 4h
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_14 = np.zeros(n)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
        
        # Sum of ATR over 14 periods
        sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
        # Max(high) - Min(low) over 14 periods
        max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
        min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
        range_14 = max_high - min_low
        
        # CHOP = 100 * log10(sum(ATR14) / (max(high) - min(low))) / log10(14)
        chop = np.full(n, np.nan)
        mask = (range_14 > 0) & (~np.isnan(sum_atr_14))
        chop[mask] = 100 * np.log10(sum_atr_14[mask] / range_14[mask]) / np.log10(14)
    
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
            np.isnan(chop[i]) or np.isnan(vol_ratio_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: CHOP < 38.2 = trending (breakout), CHOP > 61.8 = ranging (mean revert) ---
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 1.5
        
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
                # Take profit at Donchian Low (for longs) or High (for shorts)
                if close[i] <= donchian_low[i] or close[i] >= donchian_high[i]:
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
                # Take profit at Donchian High (for shorts) or Low (for longs)
                if close[i] >= donchian_high[i] or close[i] <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # In trending regime: Donchian breakout with volume
        # In ranging regime: Donchian touch fade (mean reversion)
        if is_trending:
            # Long breakout: price > Donchian High + volume
            long_condition = (close[i] > donchian_high[i]) and volume_spike
            # Short breakdown: price < Donchian Low + volume
            short_condition = (close[i] < donchian_low[i]) and volume_spike
        elif is_ranging:
            # Long mean reversion: price touches Donchian Low + volume
            long_condition = (close[i] <= donchian_low[i] * 1.001) and volume_spike
            # Short mean reversion: price touches Donchian High + volume
            short_condition = (close[i] >= donchian_high[i] * 0.999) and volume_spike
        else:
            # Choppy regime (38.2 <= CHOP <= 61.8): no trades
            long_condition = False
            short_condition = False
        
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