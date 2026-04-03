#!/usr/bin/env python3
"""
Experiment #1086: 4h Donchian(20) Breakout + HMA Trend + Volume Spike + 1d HMA Filter
HYPOTHESIS: Donchian(20) breakouts capture institutional order flow. Long when price breaks above upper band with HMA(21) uptrend and volume spike (>1.5x avg). Short when price breaks below lower band with HMA(21) downtrend and volume spike. Uses discrete position sizing (0.30) to balance reward/risk. Target: 75-200 total trades over 4 years (19-50/year) on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1086_4h_donchian20_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for additional trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    hma_1d = calculate_hma(close_1d, 21)
    hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    
    # === 4h Indicators: Donchian(20) ===
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: HMA(21) for trend filter ===
    hma = calculate_hma(close, 21)
    
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
    SIZE = 0.30  # 30% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 20  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or
            np.isnan(hma[i]) or np.isnan(hma_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
            
            # Optional: time-based exit after 8 bars (~2d on 4h) to avoid overtrading
            if bars_since_entry > 8:
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
            # HMA trend filter: 4h HMA(21) direction + 1d HMA(21) alignment
            hma_uptrend = hma[i] > hma[i-1]
            hma_downtrend = hma[i] < hma[i-1]
            hma_1d_uptrend = hma_1d_aligned[i] > hma_1d_aligned[i-1] if i > 0 else False
            hma_1d_downtrend = hma_1d_aligned[i] < hma_1d_aligned[i-1] if i > 0 else False
            
            # Breakout continuation: price breaks above upper band OR below lower band
            if price > donch_high[i] and hma_uptrend and hma_1d_uptrend:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < donch_low[i] and hma_downtrend and hma_1d_downtrend:
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