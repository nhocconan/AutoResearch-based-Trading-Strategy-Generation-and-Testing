#!/usr/bin/env python3
"""
Experiment #363: 4h Donchian(20) Breakout + 12h Volume Spike + 1d Choppiness Regime Filter

HYPOTHESIS: Donchian(20) breakouts on 4h capture significant momentum moves. 
Volume confirmation on 12h (>2.0x average) ensures institutional participation. 
Choppiness regime filter on 1d (CHOP > 61.8 = ranging, < 38.2 = trending) allows 
trend following in trending markets and mean reversion at Donchian bounds in ranging markets. 
ATR stoploss manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_vol_chop_v1"
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
    
    # === HTF: 1d data for Choppiness Index (Call ONCE before loop) ===
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
        
        # Choppiness Index formula: 100 * log10(atr_sum / (hh - ll)) / log10(14)
        # Avoid division by zero
        hl_range = hh - ll
        chop = np.full(len(atr_sum), 50.0)  # Default to neutral
        valid = (hl_range > 0) & (~np.isnan(atr_sum))
        chop[valid] = 100 * np.log10(atr_sum[valid] / hl_range[valid]) / np.log10(14)
        
        # Align to LTF
        chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    else:
        chop_aligned = np.full(n, 50.0)  # Default to neutral if insufficient data
    
    # === LTF: 4h Donchian Channel (20) ===
    donchian_window = 20
    if n >= donchian_window:
        dc_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
        dc_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
        dc_mid = (dc_high + dc_low) / 2
    else:
        dc_high = high.copy()
        dc_low = low.copy()
        dc_mid = close.copy()
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_window)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_ratio_12h_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Choppiness Index ---
        # CHOP > 61.8 = ranging market (mean reversion)
        # CHOP < 38.2 = trending market (trend follow)
        is_ranging = chop_aligned[i] > 61.8
        is_trending = chop_aligned[i] < 38.2
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio_12h_aligned[i] > 2.0
        
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
                # Take profit at Donchian midpoint or opposite band
                if close[i] >= dc_high[i] or close[i] <= dc_mid[i]:
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
                # Take profit at Donchian midpoint or opposite band
                if close[i] <= dc_low[i] or close[i] >= dc_mid[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        if is_trending and volume_spike:
            # Trending market: Donchian breakout with volume
            long_condition = close[i] > dc_high[i]
            short_condition = close[i] < dc_low[i]
        elif is_ranging:
            # Ranging market: Mean reversion at Donchian bands
            long_condition = close[i] <= dc_low[i] * 1.001  # Near lower band
            short_condition = close[i] >= dc_high[i] * 0.999  # Near upper band
        else:
            # Choppy/transition market: No trades
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