#!/usr/bin/env python3
"""
Experiment #3830: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture medium-term swings with weekly HMA (21) filtering trend direction. Volume >1.5x average confirms participation. Works in bull markets (breakouts above resistance with rising HMA) and bear markets (breakdowns below support with falling HMA). Discrete position sizing (0.25) minimizes fee drag. Target: 75-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3830_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA(21) trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate HMA(21) on weekly close
    def hull_moving_average(arr, period):
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        if half_period < 1:
            half_period = 1
        if sqrt_period < 1:
            sqrt_period = 1
        wma_half = pd.Series(arr).ewm(span=half_period, adjust=False).mean().values
        wma_full = pd.Series(arr).ewm(span=period, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma = pd.Series(raw_hma).ewm(span=sqrt_period, adjust=False).mean().values
        return hma
    
    hma_21w = hull_moving_average(close_1w, 21)
    hma_21w_aligned = align_htf_to_ltf(prices, df_1w, hma_21w)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
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
    
    warmup = max(lookback_dc + 1, 20)  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(hma_21w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price reverses through Donchian opposite band
            if position_side > 0:  # Long
                if price < lowest_low[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                if price > highest_high[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND weekly HMA rising
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                hma_21w_aligned[i] > hma_21w_aligned[i-1]):  # Weekly HMA rising
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND weekly HMA falling
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  hma_21w_aligned[i] < hma_21w_aligned[i-1]):  # Weekly HMA falling
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals