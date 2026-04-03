#!/usr/bin/env python3
"""
Experiment #290: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining daily Donchian channel breakouts with weekly HMA trend alignment and volume confirmation creates a robust trend-following strategy. The 1d Donchian(20) captures medium-term breakouts, the 1w HMA(21) filters for primary trend direction, and volume spikes confirm institutional participation. Uses discrete position sizing (0.25) and ATR-based stops to minimize fee drag while targeting 7-25 trades/year on 1d timeframe. Designed to work in both bull markets (breakouts with trend) and bear markets (short breakdowns against trend).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian20_hma_volume_v1"
timeframe = "1d"
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
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
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
            # Pad to original length
            hma_21_1w_padded = np.full(len(close_1w), np.nan)
            hma_21_1w_padded[half_len:half_len+len(hma_21_1w)] = hma_21_1w
            hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w_padded)
        else:
            hma_21_1w_aligned = np.full(n, np.nan)
    else:
        hma_21_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    # Donchian Channel(20)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    if n >= 20:
        for i in range(20-1, n):
            donchian_high[i] = np.max(high[i-20+1:i+1])
            donchian_low[i] = np.min(low[i-20+1:i+1])
    
    # Volume spike detection (volume > 2.0 * 20-period average)
    vol_ma_20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_spike = volume > (2.0 * vol_ma_20)
    else:
        vol_spike = np.zeros(n, dtype=bool)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
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
        # Long: Price breaks above Donchian(20) high + above weekly HMA + volume spike
        long_condition = (close[i] > donchian_high[i] and 
                         close[i] > hma_21_1w_aligned[i] and 
                         vol_spike[i])
        
        # Short: Price breaks below Donchian(20) low + below weekly HMA + volume spike
        short_condition = (close[i] < donchian_low[i] and 
                          close[i] < hma_21_1w_aligned[i] and 
                          vol_spike[i])
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_bar = i
            entry_price = close[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_bar = i
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals