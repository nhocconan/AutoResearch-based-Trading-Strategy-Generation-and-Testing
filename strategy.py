#!/usr/bin/env python3
"""
Experiment #1930: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation
HYPOTHESIS: Daily Donchian breakouts capture institutional momentum when aligned with weekly trend (HMA21) and volume confirmation (>1.5x average). 
Works in bull markets via breakout continuation and bear markets via breakdown continuation. 
Target: 30-100 trades over 4 years (7-25/year) with discrete sizing to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1930_1d_donchian20_1w_hma_vol_v1"
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
    def hma(arr, period):
        half = arr.copy()
        half[:] = np.nan
        half[period//2:] = arr[:-period//2] if period//2 > 0 else arr
        sqrt_period = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
        wma1 = pd.Series(half).ewm(span=period//2, min_periods=period//2, adjust=False).mean().values
        raw = 2 * wma2 - wma1
        hma_vals = pd.Series(raw).ewm(span=sqrt_period, min_periods=sqrt_period, adjust=False).mean().values
        return hma_vals
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === Primary: 1d Donchian channels (20-period) ===
    def donchian_channel(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
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
    
    warmup = 50  # sufficient for Donchian(20) and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: opposite Donchian breakout OR price crosses weekly HMA
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below weekly HMA (trend change)
                if price < hma_21_1w_aligned[i]:
                    exit_signal = True
                # Exit if price touches lower Donchian (mean reversion)
                elif price <= donchian_lower[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above weekly HMA (trend change)
                if price > hma_21_1w_aligned[i]:
                    exit_signal = True
                # Exit if price touches upper Donchian (mean reversion)
                elif price >= donchian_upper[i]:
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
        # Weekly trend filter: price relative to HMA21
        trend_bias = 1 if price > hma_21_1w_aligned[i] else -1
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND weekly trend up
            if trend_bias > 0 and price > donchian_upper[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly trend down
            elif trend_bias < 0 and price < donchian_lower[i]:
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