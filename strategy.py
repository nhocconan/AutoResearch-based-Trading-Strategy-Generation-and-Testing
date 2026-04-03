#!/usr/bin/env python3
"""
Experiment #288: 12h Donchian(20) breakout + 1w HMA(21) trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian(20) breakouts on 12h timeframe capture significant price moves with structure. 
Filtering by 1w HMA(21) ensures we trade with the higher timeframe trend, reducing whipsaws. 
Volume confirmation adds conviction to breakouts. ATR-based stoploss manages risk. 
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag while capturing 
major trend moves in both bull and bear markets. The 12h timeframe naturally filters noise 
and reduces trade frequency, improving test generalization.
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
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
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
            raw_hma = 2 * wma_half - wma_full
            hma_21_1w = wma(raw_hma, sqrt_len)
            hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
        else:
            hma_21_1w_aligned = np.full(n, np.nan)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === HTF: 1d data for volume average (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate average volume over 20 periods on 1d
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        avg_vol_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    else:
        avg_vol_20_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    if n >= 20:
        highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high_20
        donchian_lower = lowest_low_20
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # ATR(14) for stoploss
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    else:
        atr_14 = np.full(n, np.nan)
    
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
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i]) or 
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or trend reversal) ---
        if in_position:
            # Stoploss: 2 * ATR against position
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes below Donchian lower (trend weakening)
                if close[i] < donchian_lower[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price closes above Donchian upper (trend weakening)
                if close[i] > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: current volume > 1.5 * average 1d volume
        volume_confirmed = volume[i] > 1.5 * avg_vol_20_1d_aligned[i]
        
        # Long: Price breaks above Donchian upper with volume and above 1w HMA (bullish trend)
        if (close[i] > donchian_upper[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower with volume and below 1w HMA (bearish trend)
        elif (close[i] < donchian_lower[i] and 
              close[i] < hma_21_1w_aligned[i] and 
              volume_confirmed):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals