#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and volume confirmation.
Long when price breaks above upper Donchian channel AND 1w HMA is rising AND volume > 1.5x average.
Short when price breaks below lower Donchian channel AND 1w HMA is falling AND volume > 1.5x average.
ATR-based stoploss and discrete position sizing (0.25) to minimize fee churn.
Designed for low trade frequency (target: 50-100 total trades over 4 years) to work in both bull and bear markets.
"""

name = "exp_6450_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_hma(arr, period):
    """Hull Moving Average"""
    if len(arr) < period:
        return np.full_like(arr, np.nan)
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw_hma = 2 * wma2 - wma1
    hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    hma_1w = calculate_hma(df_1w['close'].values, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    hma_1w_rising = np.where(hma_1w_aligned > np.roll(hma_1w_aligned, 1), 1, 
                            np.where(hma_1w_aligned < np.roll(hma_1w_aligned, 1), -1, 0))
    
    # Precompute indicators
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    lookback = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    for i in range(lookback-1, n):
        upper_channel[i] = np.max(high[i-lookback+1:i+1])
        lower_channel[i] = np.min(low[i-lookback+1:i+1])
    
    # Average volume (20-period)
    avg_volume = np.full(n, np.nan)
    for i in range(lookback-1, n):
        avg_volume[i] = np.mean(volume[i-lookback+1:i+1])
    
    # ATR (14-period) for stoploss
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr = np.zeros(n)
    for i in range(atr_period-1, n):
        atr[i] = np.mean(tr[i-atr_period+1:i+1])
    
    # Initialize signals and position tracking
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from sufficient lookback
    start_idx = max(lookback, atr_period-1, 21)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or np.isnan(avg_volume[i]) or np.isnan(atr[i]):
            continue
            
        vol_surge = volume[i] > 1.5 * avg_volume[i]
        hma_trend = hma_1w_aligned[i]  # Use aligned HTF HMA value directly
        
        # Long entry: price breaks above upper Donchian + uptrend + volume surge
        if position <= 0 and close[i] > upper_channel[i] and hma_trend > hma_1w_aligned[i-1] and vol_surge:
            signals[i] = 0.25  # Long 25%
            position = 1
            entry_price = close[i]
        # Short entry: price breaks below lower Donchian + downtrend + volume surge
        elif position >= 0 and close[i] < lower_channel[i] and hma_trend < hma_1w_aligned[i-1] and vol_surge:
            signals[i] = -0.25  # Short 25%
            position = -1
            entry_price = close[i]
        # Stoploss and exit conditions
        elif position == 1:
            # Stoploss: 2 * ATR below entry
            if close[i] <= entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel
            elif close[i] < upper_channel[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Stoploss: 2 * ATR above entry
            if close[i] >= entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price re-enters Donchian channel
            elif close[i] > lower_channel[i]:
                signals[i] = 0.0
                position = 0
        else:
            # Hold current position
            signals[i] = signals[i-1] if i > 0 else 0.0
    
    return signals