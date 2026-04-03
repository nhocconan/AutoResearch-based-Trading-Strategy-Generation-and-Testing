#!/usr/bin/env python3
"""
Experiment #288: 12h Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation + ATR Stoploss

HYPOTHESIS: Using 12h timeframe with Donchian channel breakouts for entry, aligned with 1-week HMA trend filter, 
volume confirmation to ensure institutional participation, and ATR-based stoploss. Targets 12-37 trades/year 
(50-150 total over 4 years) to minimize fee drag while capturing medium-term trends. The 12h timeframe avoids 
overtrading pitfalls of lower timeframes while the weekly HMA ensures we only trade with the dominant trend. 
Works in both bull (breakouts above upper channel) and bear (breakouts below lower channel) markets.
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
        wma_diff = 2 * wma_half - wma_full
        hma_21_1w = wma(wma_diff, sqrt_len)
        
        # Pad beginning with NaN
        hma_21_1w_padded = np.full(len(close_1w), np.nan)
        hma_21_1w_padded[half_len + sqrt_len - 1:] = hma_21_1w[:len(hma_21_1w_padded) - (half_len + sqrt_len - 1)]
        hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 12h Indicators ===
    # Donchian Channel(20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(hma_21_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
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
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Long: Price breaks above upper Donchian + above 1w HMA + volume confirmation
        if (close[i] > highest_high[i] and 
            close[i] > hma_21_1w_aligned[i] and 
            volume_confirmed):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        
        # Short: Price breaks below lower Donchian + below 1w HMA + volume confirmation
        elif (close[i] < lowest_low[i] and 
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