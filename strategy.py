#!/usr/bin/env python3
"""
Experiment #5625: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: On 12h timeframe, Donchian(20) breakouts with volume > 2.0x average and aligned 
with 1-day HMA(21) trend capture high-probability trend continuation moves. 
The 1-day HMA filter ensures we only trade in the direction of the higher timeframe trend, 
reducing whipsaws in choppy markets. Volume confirmation validates breakout strength. 
Works in both bull and bear markets by trading breakouts in the direction of the 1d trend. 
ATR-based trailing stop (2.5x ATR) manages risk. Discrete position sizing (0.30) 
minimizes fee churn. Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5625_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for HMA(21) trend ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # Calculate HMA(21) on 1d data
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA function
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_1d = pd.Series(df_1d['close'].values)
        wma_half = wma(close_1d.values, half_len)
        wma_full = wma(close_1d.values, 21)
        wma_sqrt = wma(close_1d.values, sqrt_len)
        
        # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        hma_input = 2 * wma_half - wma_full
        hma_values = wma(hma_input, sqrt_len)
        
        # Pad beginning with NaN to match length
        hma_padded = np.full(len(close_1d), np.nan)
        hma_padded[half_len:] = hma_values[:len(close_1d) - half_len]
        hma_1d = hma_padded
    else:
        hma_1d = np.full(len(df_1d), np.nan)
    
    # Align 1d HMA to 12h timeframe
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 12h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 12h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.30  # 30% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14, 21)  # Donchian, volume avg, ATR, HMA
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (weekends) ---
        hour = hours[i]
        if hour == 0:  # Avoid first hour of day (potential gap)
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit: stoploss OR price breaks below Donchian low OR HMA turns down
                if price <= stop_price or price <= donchian_low[i] or close[i] < hma_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit: stoploss OR price breaks above Donchian high OR HMA turns up
                if price >= stop_price or price >= donchian_high[i] or close[i] > hma_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0
        
        # Trend filter: breakout in direction of 1d HMA trend
        # Long: breakout above Donchian high with price above 1d HMA (uptrend)
        # Short: breakout below Donchian low with price below 1d HMA (downtrend)
        long_setup = breakout_up and volume_confirmed and (close[i] > hma_1d_aligned[i])
        short_setup = breakout_down and volume_confirmed and (close[i] < hma_1d_aligned[i])
        
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