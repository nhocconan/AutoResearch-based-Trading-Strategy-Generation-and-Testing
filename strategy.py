#!/usr/bin/env python3
"""
Experiment #1153: 4h Donchian(20) Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 4h capture swing moves with institutional volume confirmation.
12h HMA(21) trend filter prevents counter-trend entries. Discrete position sizing (0.25) and 
ATR-based stoploss (2.0x ATR) control risk. Designed for 75-200 trades over 4 years (19-50/year).
Works in bull markets (breakouts continue) and bear markets (breakdowns continue with trend filter).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1153_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate HMA(21) on 12h: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_len = 21 // 2
    sqrt_len = int(np.sqrt(21))
    
    def wma(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights / weights.sum(), mode='valid')
    
    # Pad arrays for WMA calculation
    wma_half = np.full_like(close_12h, np.nan)
    wma_full = np.full_like(close_12h, np.nan)
    
    for i in range(half_len, len(close_12h)):
        wma_half[i] = np.mean(close_12h[i-half_len+1:i+1] * np.arange(1, half_len+1)) / (half_len*(half_len+1)/2)
    for i in range(21, len(close_12h)):
        wma_full[i] = np.mean(close_12h[i-20:i+1] * np.arange(1, 22)) / (21*22/2)
    
    # HMA = WMA(2*WMA_half - WMA_full, sqrt_len)
    raw_hma = np.full_like(close_12h, np.nan)
    for i in range(sqrt_len-1, len(close_12h)):
        if not (np.isnan(wma_half[i]) or np.isnan(wma_full[i])):
            val = 2 * wma_half[i] - wma_full[i]
            start_idx = i - sqrt_len + 1
            if start_idx >= 0:
                segment = raw_hma[start_idx:i+1] if start_idx > 0 else np.array([])
                if len(segment) == sqrt_len:
                    weights = np.arange(1, sqrt_len + 1)
                    raw_hma[i] = np.sum(segment * weights) / (sqrt_len * (sqrt_len + 1) / 2)
                elif len(segment) > 0:
                    # Handle edge case with available data
                    weights = np.arange(1, len(segment) + 1)
                    raw_hma[i] = np.sum(segment * weights) / (len(segment) * (len(segment) + 1) / 2)
    
    hma_12h = raw_hma
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Breakout: price breaks above upper band OR below lower band
            if price > donch_high[i] and close[i] > hma_12h_aligned[i]:  # 12h uptrend (price above HMA)
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and close[i] < hma_12h_aligned[i]:  # 12h downtrend (price below HMA)
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