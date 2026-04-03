#!/usr/bin/env python3
"""
Experiment #1950: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture institutional momentum when aligned with weekly trend and volume spike. 
Weekly HMA filter avoids counter-trend trades. Volume confirmation ensures breakout validity.
Works in bull/bear markets by following higher timeframe momentum. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1950_1d_donchian20_1w_hma_vol_v1"
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
    close_1w = df_1w['close'].values
    
    # Calculate weekly HMA(21)
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    # Calculate WMA for half length
    wma_half = np.full_like(close_1w, np.nan)
    for i in range(half_len - 1, len(close_1w)):
        wma_half[i] = wma(close_1w[i - half_len + 1:i + 1], half_len)
    
    # Calculate WMA for full length
    wma_full = np.full_like(close_1w, np.nan)
    for i in range(21 - 1, len(close_1w)):
        wma_full[i] = wma(close_1w[i - 21 + 1:i + 1], 21)
    
    # Calculate raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final HMA: WMA of raw with sqrt(n) length
    hma_1w = np.full_like(close_1w, np.nan)
    for i in range(sqrt_len - 1, len(raw_hma)):
        valid_raw = raw_hma[i - sqrt_len + 1:i + 1]
        if not np.any(np.isnan(valid_raw)):
            hma_1w[i] = wma(valid_raw, sqrt_len)
    
    # 1w trend: price > HMA(21) = uptrend
    trend_1w = np.where(close_1w > hma_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === Primary: 1d Donchian(20) channels ===
    # Donchian upper = max(high, lookback=20)
    # Donchian lower = min(low, lookback=20)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === Volume confirmation: 1d volume > 1.5x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = max(lookback, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price returns to Donchian lower (mean reversion)
                if price <= lowest_low[i]:
                    exit_signal = True
                # Optional: exit on opposite Donchian touch (strong reversal signal)
                elif price >= highest_high[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price returns to Donchian upper
                if price >= highest_high[i]:
                    exit_signal = True
                # Exit on opposite Donchian touch
                elif price <= lowest_low[i]:
                    exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require weekly trend alignment for bias filter
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND weekly trend up
            if trend_bias > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND weekly trend down
            elif trend_bias < 0 and price < lowest_low[i]:
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