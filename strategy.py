#!/usr/bin/env python3
"""
Experiment #248: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining 12h Donchian channel breakouts with 1-week HMA trend alignment and volume confirmation creates a robust trend-following strategy. The 1-week HMA filters for primary trend direction, Donchian(20) captures intermediate-term breakouts, and volume spike confirms institutional participation. Targets 12-37 trades/year on 12h timeframe (50-150 total over 4 years) to minimize fee drag while capturing major trend moves in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_hma_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on 1w close
    if len(df_1w) >= 21:
        close_1w = df_1w['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            hma_input = 2 * wma_half[-len(wma_full):] - wma_full
            hma_1w = wma(hma_input, sqrt_len)
            # Pad beginning with NaN
            hma_1w_full = np.full(len(close_1w), np.nan)
            hma_1w_full[21 - 1:] = hma_1w  # Adjust for WMA padding
            hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
        else:
            hma_1w_aligned = np.full(n, np.nan)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    donchian_period = 20
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    
    for i in range(n):
        start_idx = max(0, i - donchian_period + 1)
        highest_high[i] = np.max(high[start_idx:i+1])
        lowest_low[i] = np.min(low[start_idx:i+1])
    
    donchian_upper = highest_high
    donchian_lower = lowest_low
    
    # Volume Spike Detection (Volume > 2.0 * 20-period average)
    vol_ma_20 = np.zeros(n)
    for i in range(n):
        if i >= 19:  # min_periods=20
            vol_ma_20[i] = np.mean(volume[i-19:i+1])
        else:
            vol_ma_20[i] = np.nan
    volume_spike = volume > (2.0 * vol_ma_20)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    entry_bar = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1w_aligned[i]) or np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Trend Filter: 1w HMA ---
        price_above_hma = close[i] > hma_1w_aligned[i]
        price_below_hma = close[i] < hma_1w_aligned[i]
        
        # --- Breakout Conditions ---
        breakout_up = close[i] > donchian_upper[i-1]  # Break above previous period's upper band
        breakout_down = close[i] < donchian_lower[i-1]  # Break below previous period's lower band
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = np.zeros(i+1)
            atr_14[0] = tr[0]
            for j in range(1, i+1):
                atr_14[j] = (atr_14[j-1] * 13 + tr[j]) / 14  # Wilder's smoothing
            atr_value = atr_14[i]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_value
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_value
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Breakout above Donchian upper with price above 1w HMA and volume spike
        if breakout_up and price_above_hma and volume_spike[i]:
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Breakout below Donchian lower with price below 1w HMA and volume spike
        elif breakout_down and price_below_hma and volume_spike[i]:
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals