#!/usr/bin/env python3
"""
Experiment #243: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation

HYPOTHESIS: Combining 4h Donchian channel breakouts (structure) with 12h HMA trend alignment (direction) and volume confirmation (institutional participation) creates a robust trend-following strategy. The 4h timeframe minimizes fee drag while capturing intermediate-term swings. Volume confirmation filters false breakouts. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to balance statistical significance with low fee impact. Works in both bull (breakouts continuation) and bear (breakdown continuation) markets via symmetric long/short logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_12h_v1"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        wma_half = pd.Series(close_12h).rolling(window=half_len, min_periods=half_len).mean().values
        wma_full = pd.Series(close_12h).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_21_12h = pd.Series(raw_hma).rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    else:
        hma_21_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_len = 20
    if n >= donchian_len:
        dc_upper = pd.Series(high).rolling(window=donchian_len, min_periods=donchian_len).max().values
        dc_lower = pd.Series(low).rolling(window=donchian_len, min_periods=donchian_len).min().values
    else:
        dc_upper = np.full(n, np.nan)
        dc_lower = np.full(n, np.nan)
    
    # Volume filter: volume > 1.5x 20-period average
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (vol_ma * 1.5)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = max(100, donchian_len)  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or np.isnan(hma_21_12h_aligned[i])):
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
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper + above 12h HMA + volume confirmation
        if (close[i] > dc_upper[i] and 
            close[i] > hma_21_12h_aligned[i] and 
            volume_filter[i]):
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower + below 12h HMA + volume confirmation
        elif (close[i] < dc_lower[i] and 
              close[i] < hma_21_12h_aligned[i] and 
              volume_filter[i]):
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals