#!/usr/bin/env python3
"""
Experiment #1031: 6h Donchian(20) Breakout + 1d Camarilla Pivot + Volume Confirmation
HYPOTHESIS: Combining 6h Donchian breakouts with 1d Camarilla pivot levels creates high-probability entries. 
In ranging markets, price tends to reverse at R3/S3 and S4/R4 levels. In trending markets, breaks of R4/S4 
with volume confirmation capture continuation. Uses 1d HTF for pivot calculation to avoid look-ahead. 
Target: 75-150 total trades over 4 years (19-38/year) on 6h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1031_6h_donchian20_1d_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    camarilla_r4, camarilla_r3, camarilla_s3, camarilla_s4 = calculate_camarilla(high_1d, low_1d, close_1d)
    
    # Align HTF Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # === 6h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i])):
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
            
            # Optional: time-based exit after 12 bars (~3d on 6h) to avoid overtrading
            if bars_since_entry > 12:
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
            # Camarilla-based entry logic
            r4 = r4_aligned[i]
            r3 = r3_aligned[i]
            s3 = s3_aligned[i]
            s4 = s4_aligned[i]
            
            # Long conditions:
            # 1. Break above R4 with volume (bullish continuation)
            # 2. Rejection at S3 with volume (mean reversion long)
            long_breakout = price > donch_high[i] and price > r4
            long_rejection = price < donch_low[i] and price > s3 and price < s4
            
            # Short conditions:
            # 1. Break below S4 with volume (bearish continuation)
            # 2. Rejection at R3 with volume (mean reversion short)
            short_breakout = price < donch_low[i] and price < s4
            short_rejection = price > donch_high[i] and price < r3 and price > r4
            
            if long_breakout or long_rejection:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif short_breakout or short_rejection:
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

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    n = len(high)
    r4 = np.full(n, np.nan)
    r3 = np.full(n, np.nan)
    s3 = np.full(n, np.nan)
    s4 = np.full(n, np.nan)
    
    for i in range(n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i]):
            continue
        diff = high[i] - low[i]
        c = close[i]
        r4[i] = c + (diff * 1.1 / 2)
        r3[i] = c + (diff * 1.1 / 4)
        s3[i] = c - (diff * 1.1 / 4)
        s4[i] = c - (diff * 1.1 / 2)
    
    return r4, r3, s3, s4

def calculate_hma(close, period):
    """Calculate Hull Moving Average"""
    if len(close) < period:
        return np.full_like(close, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA for half period
    wma_half = np.zeros_like(close)
    for i in range(half_period, len(close)):
        weights = np.arange(1, half_period + 1)
        wma_half[i] = np.dot(close[i - half_period + 1:i + 1], weights) / weights.sum()
    
    # WMA for full period
    wma_full = np.zeros_like(close)
    for i in range(period, len(close)):
        weights = np.arange(1, period + 1)
        wma_full[i] = np.dot(close[i - period + 1:i + 1], weights) / weights.sum()
    
    # Raw HMA: 2*WMA(half) - WMA(full)
    raw_hma = 2 * wma_half - wma_full
    
    # Final WMA of raw HMA with sqrt_period
    hma = np.zeros_like(close)
    for i in range(sqrt_period, len(close)):
        weights = np.arange(1, sqrt_period + 1)
        hma[i] = np.dot(raw_hma[i - sqrt_period + 1:i + 1], weights) / weights.sum()
    
    return hma