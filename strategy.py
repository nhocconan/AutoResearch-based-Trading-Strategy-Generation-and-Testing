#!/usr/bin/env python3
"""
Experiment #040: 4h Donchian(20) Breakout + 1d HMA21 Trend + Volume Spike
HYPOTHESIS: 4h Donchian breakouts capture momentum bursts, filtered by 1d HMA21 trend direction to avoid counter-trend whipsaws. Volume spikes (>2x MA20) confirm institutional participation. This structure works in both bull/bear markets by only taking breakouts in alignment with the higher timeframe trend. Uses discrete sizing (0.25) and ATR(14) stoploss (2.0x) to manage risk. Target: 100-200 total trades over 4 years (25-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_040_4h_donchian20_1d_hma21_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA21 trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    hma_21 = calculate_hma(df_1d['close'].values, 21)
    hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    
    # === 4h Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    valid_start = 20
    vol_ratio[valid_start:] = volume[valid_start:] / vol_ma[valid_start:]
    vol_ratio[:valid_start] = 1.0
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian20 and HMA21
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_21_aligned[i]) or np.isnan(highest_20[i]) or 
            np.isnan(lowest_20[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Trend Filter: 1d HMA21 ---
        bullish_trend = price > hma_21_aligned[i]
        bearish_trend = price < hma_21_aligned[i]
        
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
            
            # Optional: time-based exit after 6 bars (~1d on 4h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Donchian breakout in direction of 1d HMA21 trend
            if price > highest_20[i] and bullish_trend:  # Break above upper band with bullish trend
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            elif price < lowest_20[i] and bearish_trend:  # Break below lower band with bearish trend
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

def calculate_hma(values, period):
    """Calculate Hull Moving Average"""
    values = pd.Series(values)
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = values.ewm(span=half_period, adjust=False).mean()
    wma_full = values.ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma_half - wma_full
    hma = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
    
    return hma.values