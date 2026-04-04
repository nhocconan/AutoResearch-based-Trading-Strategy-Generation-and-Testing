#!/usr/bin/env python3
"""
Experiment #6353: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation
HYPOTHESIS: 4h Donchian breakouts with volume confirmation (>1.5x avg) and 12h HMA trend filter capture institutional momentum while avoiding whipsaws. 
In bull markets, HMA uptrend confirms breakout validity. In bear markets, HMA downtrend filters false breakouts. 
Volume confirmation ensures breakouts have participation. Uses discrete sizing (0.25) to minimize fee churn. 
Target: 75-200 trades over 4 years (19-50/year). Works in both bull and bear via trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6353_4h_donchian20_12h_hma_vol_v1"
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
    
    # === HTF: 12h data for HMA trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Hull Moving Average (HMA) calculation
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(values, period):
            if len(values) < period:
                return np.full_like(values, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_12h = df_12h['close'].values
        n_period = 21
        half_n = n_period // 2
        sqrt_n = int(np.sqrt(n_period))
        
        wma_half = wma(close_12h, half_n)
        wma_full = wma(close_12h, n_period)
        
        # 2*WMA(half) - WMA(full)
        wma_diff = 2 * wma_half - wma_full
        
        # WMA of the difference with sqrt(n) period
        hma_values = wma(wma_diff, sqrt_n)
        
        # Pad to original length
        hma_padded = np.full(len(close_12h), np.nan)
        start_idx = len(close_12h) - len(hma_values)
        if start_idx >= 0:
            hma_padded[start_idx:] = hma_values
        
        # Align to 4h timeframe
        hma_aligned = align_htf_to_ltf(prices, df_12h, hma_padded)
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
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
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
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. HMA trend turns down (exit long when trend deteriorates)
                if price <= stop_price or price <= donchian_low[i] or hma_aligned[i] < hma_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. HMA trend turns up (exit short when trend deteriorates)
                if price >= stop_price or price >= donchian_high[i] or hma_aligned[i] > hma_aligned[i-1]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 1.5  # Volume filter
        hma_uptrend = hma_aligned[i] > hma_aligned[i-1]
        hma_downtrend = hma_aligned[i] < hma_aligned[i-1]
        
        long_entry = breakout_up and volume_confirmed and hma_uptrend
        short_entry = breakout_down and volume_confirmed and hma_downtrend
        
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
    
    return signals