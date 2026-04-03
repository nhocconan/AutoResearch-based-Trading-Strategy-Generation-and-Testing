#!/usr/bin/env python3
"""
Experiment #273: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves, especially when aligned with higher timeframe trend (12h HMA) and confirmed by volume spikes. The ATR-based stoploss limits drawdowns during reversals. This structure works in both bull and bear markets by trading breakouts in the direction of the 12h trend. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_12h_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate HMA(21) on 12h close
    if len(df_12h) >= 21:
        close_12h = df_12h['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        n_hma = 21
        half_n = n_hma // 2
        sqrt_n = int(np.sqrt(n_hma))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_12h, half_n)
        wma_full = wma(close_12h, n_hma)
        raw_hma = 2 * wma_half - wma_full
        hma_12h = wma(raw_hma, sqrt_n)
        
        # Pad to match original length
        hma_12h_padded = np.full(len(close_12h), np.nan)
        hma_12h_padded[half_n + sqrt_n - 1:] = hma_12h
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_padded)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_period = 20
    highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    # Volume Spike: volume > 1.5 * 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * avg_volume)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_period)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_12h_aligned[i]) or np.isnan(avg_volume[i])):
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
        # Long: Price breaks above Donchian upper band + above 12h HMA + volume spike
        if (close[i] > highest_high[i] and 
            close[i] > hma_12h_aligned[i] and 
            volume_spike[i]):
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower band + below 12h HMA + volume spike
        elif (close[i] < lowest_low[i] and 
              close[i] < hma_12h_aligned[i] and 
              volume_spike[i]):
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals