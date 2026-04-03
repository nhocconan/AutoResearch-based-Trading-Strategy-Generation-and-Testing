#!/usr/bin/env python3
"""
Experiment #764: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture significant momentum, filtered by weekly HMA trend 
and volume spikes (>1.5x average). Works in both bull and bear markets because: 
- In bull markets, price breaks above upper Donchian with HMA up and volume confirmation → long
- In bear markets, price breaks below lower Donchian with HMA down and volume confirmation → short
- Weekly HMA ensures we only trade with the higher timeframe trend, reducing whipsaws
- Volume confirmation ensures breakouts have conviction
Target: 30-100 total trades over 4 years (7-25/year) on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_764_1d_donchian20_1w_hma_vol_v1"
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
    
    # Calculate HMA(21) on weekly
    def calculate_hma(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1:
            half_period = 1
        if sqrt_period < 1:
            sqrt_period = 1
        wma2 = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma2 - wma1
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_1w = calculate_hma(close_1w, 21)
    # HMA trend: 1 = rising (current > previous), 0 = falling
    hma_trend_1w = np.where(hma_1w > np.roll(hma_1w, 1), 1, 0)
    hma_trend_1w[0] = 0  # first value undefined
    # Align HMA trend to daily timeframe
    hma_trend_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_trend_1w)
    
    # === 1d Indicators: Donchian Channel(20) ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
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
    
    warmup = max(lookback, 20)  # sufficient for Donchian and volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic: Opposite Donchian break ---
        if in_position:
            if position_side > 0:  # Long position
                # Exit if price breaks below lower Donchian
                if low[i] < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Exit if price breaks above upper Donchian
                if high[i] > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike and hma_trend_1w_aligned[i] > 0:  # Only trade with weekly HMA trend
            # Long: price breaks above upper Donchian AND weekly HMA trending up
            if high[i] > highest_high[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short: price breaks below lower Donchian AND weekly HMA trending down
            elif low[i] < lowest_low[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals

}