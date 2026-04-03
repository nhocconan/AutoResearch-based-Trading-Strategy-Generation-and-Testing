#!/usr/bin/env python3
"""
Experiment #1925: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Spike
HYPOTHESIS: Donchian channel breakouts on 12h timeframe capture significant moves, filtered by 1d HMA trend and volume confirmation. 
This avoids overtrading by requiring strong breakouts with confluence, targeting 50-150 total trades over 4 years.
Works in bull/bear markets by following the higher timeframe trend with momentum confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1925_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d HMA(21)
    def calculate_hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        if half == 0 or sqrt == 0:
            return arr.copy()
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean().values
        wma1 = pd.Series(2 * wma2 - pd.Series(arr).ewm(span=period, adjust=False).mean()).ewm(span=sqrt, adjust=False).mean().values
        return wma1
    
    hma_1d = calculate_hma(close_1d, 21)
    trend_1d = np.where(close_1d > hma_1d, 1, -1)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # === 12h Indicators: Donchian(20) channels ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
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
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss (2x ATR) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            tr1 = high[i] - low[i]
            tr2 = np.abs(high[i] - close[i-1])
            tr3 = np.abs(low[i] - close[i-1])
            tr = max(tr1, tr2, tr3)
            # Simplified: use fixed ATR approximation for exit (can be improved)
            atr_approx = np.mean([tr1, tr2, tr3])  # placeholder
            
            exit_signal = False
            if position_side > 0:  # Long position
                if price < entry_price - 2.0 * atr_approx:
                    exit_signal = True
            else:  # Short position
                if price > entry_price + 2.0 * atr_approx:
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
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 1d trend up
            if trend_1d_aligned[i] > 0 and price > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 1d trend down
            elif trend_1d_aligned[i] < 0 and price < lowest_low[i]:
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