#!/usr/bin/env python3
"""
Experiment #4337: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike
HYPOTHESIS: Donchian(20) breakouts capture institutional entry/exit points. 1d HMA(21) filter ensures alignment with daily trend to avoid counter-trend whipsaws. Volume spike (>2.0x average) confirms breakout validity. Works in bull via upside breakouts with volume, in bear via downside breakouts with volume. ATR(14) trailing stop (2.5x) manages risk. Targets 75-200 total trades over 4 years (19-50/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4337_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_1d = df_1d['close'].values.astype(np.float64)
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        wma_half = wma(close_1d, half_len)
        wma_full = wma(close_1d, 21)
        wma_diff = 2 * wma_half - wma_full
        hma_1d = wma(wma_diff, sqrt_len)
        
        # Pad to original length
        hma_padded = np.full(len(close_1d), np.nan)
        start_idx = 21 - half_len + sqrt_len - 1
        end_idx = start_idx + len(hma_1d)
        if end_idx <= len(close_1d) and start_idx >= 0:
            hma_padded[start_idx:end_idx] = hma_1d
        
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_padded)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    donchian_window = 20
    highest = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
    lowest = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(donchian_window, 20, 14, 21)  # Donchian, vol MA, ATR, HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest[i]) or np.isnan(lowest[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter: only trade when price is above/below 1d HMA
        price_above_hma = price > hma_1d_aligned[i]
        price_below_hma = price < hma_1d_aligned[i]
        
        if volume_confirm:
            # Long conditions: Break above Donchian upper + price > 1d HMA + volume
            long_entry = (price > highest[i-1]) and price_above_hma
            
            # Short conditions: Break below Donchian lower + price < 1d HMA + volume
            short_entry = (price < lowest[i-1]) and price_below_hma
            
            if long_entry:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif short_entry:
                in_position = True
                position_side = -1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals