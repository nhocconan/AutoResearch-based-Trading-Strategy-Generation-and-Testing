#!/usr/bin/env python3
"""
Experiment #121: 4h Donchian Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian(20) breakouts on 4h timeframe, filtered by 1d HMA(21) trend direction and 
confirmed by volume spike (>2x average), capture high-probability trend continuation moves. 
The 1d HMA filter ensures alignment with higher timeframe direction, reducing false breakouts 
in choppy markets. Volume spike confirms institutional participation. Targets 19-50 trades/year 
on 4h timeframe (75-200 total over 4 years) to minimize fee drag while maintaining statistical 
validity. Works in both bull (breakouts with volume) and bear (breakdowns with volume) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_breakout_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(close_1d).ewm(span=21, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_1d = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    else:
        hma_21_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for regime filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate ADX(14) on 1w for regime filtering
    if len(df_1w) >= 14:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # True Range
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First period
        
        # Directional Movement
        dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                          np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
        dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                           np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
        dm_plus[0] = 0
        dm_minus[0] = 0
        
        # Smoothed values
        atr_1w = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False).mean().values
        dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False).mean().values
        
        # DI+ and DI-
        di_plus = 100 * dm_plus_smooth / (atr_1w + 1e-10)
        di_minus = 100 * dm_minus_smooth / (atr_1w + 1e-10)
        
        # DX and ADX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
        adx_1w = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        adx_1w_aligned = align_htf_to_ltf(prices, df_1w, adx_1w)
    else:
        adx_1w_aligned = np.full(n, 20.0)  # Neutral regime
    
    # === 4h Indicators ===
    # Donchian Channel (20)
    donchian_high = np.zeros(n)
    donchian_low = np.zeros(n)
    for i in range(n):
        start_idx = max(0, i - 19)
        donchian_high[i] = np.max(high[start_idx:i+1])
        donchian_low[i] = np.min(low[start_idx:i+1])
    
    # Volume ratio (current vs 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_1d_aligned[i]) or np.isnan(adx_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Regime Filter: Only trade when ADX > 20 (trending market) ---
        trending_market = adx_1w_aligned[i] > 20
        
        # --- Trend Filter: Price > 1d HMA21 for long, < for short ---
        price_above_1d_hma = close[i] > hma_21_1d_aligned[i]
        price_below_1d_hma = close[i] < hma_21_1d_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Exit Logic (ATR-based stoploss and trailing stop) ---
        if in_position:
            # Update highest/lowest since entry
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                # Stoploss: 2.5 * ATR below entry
                stop_level = entry_price - 2.5 * atr_14[i]
                # Trailing stop: 2.0 * ATR below highest point
                trail_stop = highest_since_entry - 2.0 * atr_14[i]
                effective_stop = max(stop_level, trail_stop)
                
                if low[i] < effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian breakout in opposite direction
                if close[i] < donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Stoploss: 2.5 * ATR above entry
                stop_level = entry_price + 2.5 * atr_14[i]
                # Trailing stop: 2.0 * ATR above lowest point
                trail_stop = lowest_since_entry + 2.0 * atr_14[i]
                effective_stop = min(stop_level, trail_stop)
                
                if high[i] > effective_stop:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian breakout in opposite direction
                if close[i] > donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + volume spike + price > 1d HMA
        long_condition = (
            close[i] > donchian_high[i] and 
            volume_spike and 
            price_above_1d_hma and
            trending_market
        )
        
        # Short: Price breaks below Donchian low + volume spike + price < 1d HMA
        short_condition = (
            close[i] < donchian_low[i] and 
            volume_spike and 
            price_below_1d_hma and
            trending_market
        )
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals