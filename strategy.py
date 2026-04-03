#!/usr/bin/env python3
"""
Experiment #1878: 1d Donchian(20) Breakout + 1w HMA(21) Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian breakouts capture sustained moves, filtered by weekly HMA trend to avoid counter-trend trades. Volume confirmation ensures breakout validity. Works in both bull/bear markets by following weekly trend. Target: 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1878_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # 1w HMA(21) for trend direction
    def hma(series, period):
        if len(series) < period:
            return np.full_like(series, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(series).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(series).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        hma_val = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
        return hma_val.values
    
    hma_21_1w = hma(close_1w, 21)
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite Donchian break or trend flip ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below Donchian low OR weekly trend flips bearish
                if price < lowest_20[i] or trend_1w_aligned[i] < 0:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above Donchian high OR weekly trend flips bullish
                if price > highest_20[i] or trend_1w_aligned[i] > 0:
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
        # Require weekly trend alignment for bias
        trend_bias = trend_1w_aligned[i]
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Donchian breakout signals
            if trend_bias > 0 and price > highest_20[i]:
                # Bullish breakout in bullish weekly trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif trend_bias < 0 and price < lowest_20[i]:
                # Bearish breakout in bearish weekly trend
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