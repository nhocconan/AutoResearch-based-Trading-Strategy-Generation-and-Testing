#!/usr/bin/env python3
"""
Experiment #764: 1d Donchian Breakout + 1w HMA Trend + Volume Spike
HYPOTHESIS: Daily Donchian(20) breakouts capture strong momentum, filtered by 1-week HMA(21) trend 
and volume confirmation (>2.0x average). Long when price breaks above Donchian upper AND 
weekly HMA rising. Short when price breaks below Donchian lower AND weekly HMA falling. 
Uses discrete position sizing (0.25) and ATR-based stoploss. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_764_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on weekly
    def calculate_hma(arr, period):
        half_len = period // 2
        sqrt_len = int(np.sqrt(period))
        if half_len == 0:
            half_len = 1
        if sqrt_len == 0:
            sqrt_len = 1
        # WMA calculation
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, 'valid') / weights.sum()
        # Handle edge cases for small arrays
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        wma_half = np.array([wma(arr[i:i+half_len], half_len) if i+half_len <= len(arr) else np.nan 
                            for i in range(len(arr))])
        wma_full = np.array([wma(arr[i:i+period], period) if i+period <= len(arr) else np.nan 
                            for i in range(len(arr))])
        wma_sqrt = np.array([wma(arr[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(arr) else np.nan 
                            for i in range(len(arr))])
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        raw_hma = 2 * wma_half - wma_full
        hma = np.array([wma(raw_hma[i:i+sqrt_len], sqrt_len) if i+sqrt_len <= len(raw_hma) else np.nan 
                       for i in range(len(raw_hma))])
        # Pad beginning with NaN
        hma = np.concatenate([np.full(sqrt_len-1, np.nan), hma[:len(arr)-sqrt_len+1]])
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    # HMA rising: 1 if current > previous, 0 otherwise
    hma_rising = np.zeros_like(hma_1w)
    hma_rising[1:] = np.where(hma_1w[1:] > hma_1w[:-1], 1, 0)
    hma_falling = np.zeros_like(hma_1w)
    hma_falling[1:] = np.where(hma_1w[1:] < hma_1w[:-1], 1, 0)
    # Align to daily timeframe
    hma_rising_aligned = align_htf_to_ltf(prices, df_1w, hma_rising)
    hma_falling_aligned = align_htf_to_ltf(prices, df_1w, hma_falling)
    
    # === 1d Indicators: Donchian Channel (20) ===
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(20, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_rising_aligned[i]) or
            np.isnan(hma_falling_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 25 bars (~25 days) to avoid overtrading
            if bars_since_entry > 25:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 2.0x average)
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Long: Price breaks above Donchian upper AND weekly HMA rising
            if price > highest_high[i] and hma_rising_aligned[i] > 0:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Price breaks below Donchian lower AND weekly HMA falling
            elif price < lowest_low[i] and hma_falling_aligned[i] > 0:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals