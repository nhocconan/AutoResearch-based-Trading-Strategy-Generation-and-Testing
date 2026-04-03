#!/usr/bin/env python3
"""
Experiment #283: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Donchian channel breakouts on 4h capture significant price moves with institutional participation. 
The 12h HMA filter ensures we only trade in the direction of the medium-term trend, reducing whipsaws. 
Volume spike confirmation (volume > 1.5x 20-period average) adds conviction that breakouts are genuine. 
ATR-based stoploss manages risk. Targets 25-50 trades/year on 4h (100-200 total over 4 years) to minimize fee drag 
while capturing high-probability trend continuations. Works in both bull and bear markets by trading breakouts 
in the direction of the 12h trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian20_hma12_volume_vspike_v1"
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
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_12h, half_len)
        wma_full = wma(close_12h, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            hma_raw = 2 * wma_half - wma_full
            hma_21_12h = wma(hma_raw, sqrt_len)
            # Pad to original length
            hma_21_12h_padded = np.full(len(close_12h), np.nan)
            start_idx = 21 - 1  # Approximate offset due to WMA padding
            if start_idx < len(hma_21_12h_padded) and len(hma_21_12h) > start_idx:
                end_idx = min(start_idx + len(hma_21_12h), len(hma_21_12h_padded))
                hma_21_12h_padded[start_idx:end_idx] = hma_21_12h[:end_idx-start_idx]
            hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h_padded)
        else:
            hma_21_12h_aligned = np.full(n, np.nan)
    else:
        hma_21_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    donchian_period = 20
    if n >= donchian_period:
        highest_high = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
        lowest_low = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
    
    # Volume Spike: volume > 1.5x 20-period average
    if n >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (vol_ma_20 * 1.5)
    else:
        volume_spike = np.zeros(n, dtype=bool)
    
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
    
    warmup = max(100, donchian_period, 20)  # Ensure enough data for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(hma_21_12h_aligned[i]) or np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Exit if price breaks below Donchian low (trend weakening)
                if close[i] < lowest_low[i]:
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
                # Exit if price breaks above Donchian high (trend weakening)
                if close[i] > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian high + volume spike + price above 12h HMA (bullish trend)
        if (close[i] > highest_high[i] and 
            volume_spike[i] and 
            close[i] > hma_21_12h_aligned[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian low + volume spike + price below 12h HMA (bearish trend)
        elif (close[i] < lowest_low[i] and 
              volume_spike[i] and 
              close[i] < hma_21_12h_aligned[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals