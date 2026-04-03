#!/usr/bin/env python3
"""
Experiment #1938: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian(20) breakouts capture institutional intraday/weekly momentum. 
Weekly HMA(21) filter ensures alignment with primary trend to avoid counter-trend whipsaws. 
Volume confirmation (>1.5x 20-period average) filters low-conviction breakouts. 
ATR-based stoploss (2.5x ATR) manages risk. Target: 75-150 total trades over 4 years.
Works in bull/bear markets by following weekly trend with daily execution precision.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1938_1d_donchian20_1w_hma_vol_v1"
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
    
    # Calculate 1w HMA(21): Weighted Moving Average of WMA
    def wma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights / weights.sum(), mode='valid')
    
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        if half < 1 or sqrt < 1:
            return np.full_like(arr, np.nan)
        wma_half = wma(arr, half)
        wma_full = wma(arr, period)
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        raw = 2 * wma_half - wma_full
        return wma(raw, sqrt)
    
    # Pad WMA/HMA results to original length
    hma_21_1w = np.full_like(close_1w, np.nan)
    hma_values = hma(close_1w, 21)
    if len(hma_values) > 0:
        hma_21_1w[20:20+len(hma_values)] = hma_values  # Account for WMA padding
    
    # 1w trend: 1 if close > HMA, -1 otherwise
    trend_1w = np.where(close_1w > hma_21_1w, 1, -1)
    trend_1w_aligned = align_htf_to_ltf(prices, df_1w, trend_1w)
    
    # === 1d Indicators: Donchian(20) channels ===
    def donchian_channels(high, low, period):
        upper = pd.Series(high).rolling(window=period, min_periods=period).max().values
        lower = pd.Series(low).rolling(window=period, min_periods=period).min().values
        return upper, lower
    
    upper_20, lower_20 = donchian_channels(high, low, 20)
    
    # === 1d Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
    def atr(high, low, close, period):
        tr1 = pd.Series(high).rolling(window=1).max() - pd.Series(low).rolling(window=1).min()
        tr2 = abs(pd.Series(high).rolling(window=1).max() - pd.Series(close).rolling(window=1).shift())
        tr3 = abs(pd.Series(low).rolling(window=1).min() - pd.Series(close).rolling(window=1).shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period, min_periods=period).mean().values
        return atr
    
    atr_14 = atr(high, low, close, 14)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    stop_price = 0.0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(trend_1w_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr_14[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: stoploss hit
            exit_signal = False
            if position_side > 0:  # Long position
                if price <= stop_price:
                    exit_signal = True
            else:  # Short position
                if price >= stop_price:
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
            # Long entry: price breaks above upper Donchian AND weekly trend up
            if trend_bias > 0 and price > upper_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                stop_price = entry_price - 2.5 * atr_14[i]
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly trend down
            elif trend_bias < 0 and price < lower_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                stop_price = entry_price + 2.5 * atr_14[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals