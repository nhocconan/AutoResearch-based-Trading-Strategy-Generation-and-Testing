#!/usr/bin/env python3
"""
Experiment #280: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: Combining 4h Donchian channel breakouts with 1d HMA trend alignment and volume confirmation creates a robust trend-following strategy that works in both bull and bear markets. The 1d HMA provides higher-timeframe trend direction to avoid counter-trend trades, volume spike confirms institutional participation, and ATR-based stoploss manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) to minimize fee drag while capturing strong directional moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        if len(wma_half) > 0 and len(wma_full) > 0:
            raw_hma = 2 * wma_half - wma_full
            hma_1d = wma(raw_hma, sqrt_len)
            # Pad to original length
            hma_1d_padded = np.full(len(close_1d), np.nan)
            hma_1d_padded[half_len:] = hma_1d[:len(close_1d)-half_len]
            hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d_padded)
        else:
            hma_1d_aligned = np.full(n, np.nan)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20)
    if n >= 20:
        highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        donchian_upper = highest_high
        donchian_lower = lowest_low
    else:
        donchian_upper = np.full(n, np.nan)
        donchian_lower = np.full(n, np.nan)
    
    # Volume Spike: volume > 1.5 * 20-period average
    if n >= 20:
        vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_spike = volume > (1.5 * vol_ma)
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
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(hma_1d_aligned[i]) or np.isnan(atr_14[i])):
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
                # Exit on Donchian lower break (trailing stop)
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
                # Exit on Donchian upper break (trailing stop)
                if close[i] > donchian_upper[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Price breaks above Donchian upper + above 1d HMA + volume spike
        if (close[i] > donchian_upper[i] and 
            close[i] > hma_1d_aligned[i] and 
            volume_spike[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            entry_bar = i
            signals[i] = SIZE
        # Short: Price breaks below Donchian lower + below 1d HMA + volume spike
        elif (close[i] < donchian_lower[i] and 
              close[i] < hma_1d_aligned[i] and 
              volume_spike[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>