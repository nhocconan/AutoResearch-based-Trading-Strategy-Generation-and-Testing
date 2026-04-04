#!/usr/bin/env python3
"""
Experiment #5897: 4h Donchian(20) breakout + 1d HMA(21) trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts aligned with 1d HMA trend direction capture high-probability 
continuation moves. Volume confirmation filters weak breakouts. HMA(21) on daily timeframe provides 
smooth trend filter that works in both bull and bear markets by reducing whipsaw. Target: 75-200 
total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5897_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for HMA(21) trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # Calculate HMA(21) on daily close prices
        # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values
        half_len = len(close_1d) // 2
        sqrt_len = int(np.sqrt(len(close_1d)))
        
        wma_half = wma(close_1d, half_len) if half_len >= 1 else np.full_like(close_1d, np.nan)
        wma_full = wma(close_1d, len(close_1d)) if len(close_1d) >= 1 else np.full_like(close_1d, np.nan)
        
        # Handle array alignment for WMA calculations
        if len(wma_half) > 0 and len(wma_full) > 0:
            # Pad wma_half to match close_1d length
            wma_half_padded = np.full_like(close_1d, np.nan)
            wma_half_padded[half_len-1:half_len-1+len(wma_half)] = wma_half
            
            # Pad wma_full to match close_1d length
            wma_full_padded = np.full_like(close_1d, np.nan)
            wma_full_padded[len(close_1d)-1:len(close_1d)-1+len(wma_full)] = wma_full
            
            # 2 * WMA(half) - WMA(full)
            diff = 2 * wma_half_padded - wma_full_padded
            
            # WMA of diff with sqrt(n) period
            if sqrt_len >= 1 and len(diff) >= sqrt_len:
                wma_diff = wma(diff, sqrt_len)
                hma_1d = np.full_like(close_1d, np.nan)
                hma_1d[sqrt_len-1:sqrt_len-1+len(wma_diff)] = wma_diff
            else:
                hma_1d = np.full_like(close_1d, np.nan)
        else:
            hma_1d = np.full_like(close_1d, np.nan)
        
        # Align to LTF (4h) with shift(1) for completed bars only
        hma_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 20)  # Donchian, volume avg, ATR
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods ---
        hour = hours[i]
        if 21 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low (failed breakout)
                if price <= stop_price or price <= donchian_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high (failed breakout)
                if price >= stop_price or price >= donchian_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5
        
        # HMA trend filter: price above HMA = uptrend, below = downtrend
        uptrend = price > hma_aligned[i]
        downtrend = price < hma_aligned[i]
        
        # Entry conditions: breakout in direction of HMA trend
        long_setup = breakout_up and volume_confirmed and uptrend
        short_setup = breakout_down and volume_confirmed and downtrend
        
        if long_setup:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_setup:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals