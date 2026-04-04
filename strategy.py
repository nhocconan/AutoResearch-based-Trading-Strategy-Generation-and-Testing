#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Works in both bull and bear markets because:
- Donchian breakouts capture strong momentum moves
- 1w HMA ensures we only trade with the higher timeframe trend
- Volume confirmation reduces false breakouts
- ATR-based stoploss manages risk
Target: 30-100 trades over 4 years (7-25/year) on 1d timeframe.
"""

name = "exp_6458_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    half = period // 2
    sqrt = int(np.sqrt(period))
    wma2 = np.convolve(arr, np.ones(half)/half, mode='same')
    wma1 = np.convolve(arr, np.ones(period)/period, mode='same')
    raw = 2 * wma2 - wma1
    hma = np.convolve(raw, np.ones(sqrt)/sqrt, mode='same')
    # Fill edges
    hma[:half] = hma[half]
    hma[-sqrt:] = hma[-sqrt-1] if len(hma) > sqrt else hma[0]
    return hma

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
    # Pre-calculate indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(lookback-1, n):
        highest_high[i] = np.max(high[i-lookback+1:i+1])
        lowest_low[i] = np.min(low[i-lookback+1:i+1])
    
    # Average True Range for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros(n)
    for i in range(1, n):
        if i < 14:
            atr[i] = np.nan
        else:
            atr[i] = np.nanmean(tr[i-13:i+1])
    
    # Volume average (20-period)
    vol_ma = np.zeros(n)
    for i in range(n):
        if i < 19:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from sufficient lookback
    start_idx = max(lookback, 21)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(hma_1w_aligned[i]):
            continue
            
        # Volume confirmation: current volume > 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below 1w HMA
        price_above_hma = close[i] > hma_1w_aligned[i]
        price_below_hma = close[i] < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        long_breakout = close[i] > highest_high[i] and vol_confirm and price_above_hma
        short_breakout = close[i] < lowest_low[i] and vol_confirm and price_below_hma
        
        # Exit conditions
        exit_long = position == 1 and (close[i] < lowest_low[i] or close[i] < entry_price - 2.5 * atr[i])
        exit_short = position == -1 and (close[i] > highest_high[i] or close[i] > entry_price + 2.5 * atr[i])
        
        # Generate signals
        if position == 0:
            if long_breakout:
                signals[i] = 0.30  # Long 30%
                position = 1
                entry_price = close[i]
            elif short_breakout:
                signals[i] = -0.30  # Short 30%
                position = -1
                entry_price = close[i]
        elif position == 1:
            if exit_long:
                signals[i] = 0.0  # Exit long
                position = 0
            else:
                signals[i] = 0.30  # Hold long
        elif position == -1:
            if exit_short:
                signals[i] = 0.0  # Exit short
                position = 0
            else:
                signals[i] = -0.30  # Hold short
    
    return signals