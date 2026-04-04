#!/usr/bin/env python3
"""
Experiment #3818: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian breakouts capture medium-term swings with 1-week HMA (21) confirming the primary trend direction and 1d volume (>1.8x) confirming institutional participation. Works in bull markets (breakouts above resistance in uptrend) and bear markets (breakdowns below support in downtrend). Discrete position sizing (0.25) minimizes fee drag. Target: 30-100 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3818_1d_donchian20_1w_hma_vol_v1"
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
    def calculate_hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        # WMA function
        def wma(values, window):
            if len(values) < window:
                return np.full_like(values, np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        wma_half = wma(arr, half_period)
        wma_full = wma(arr, period)
        
        if len(wma_half) == 0 or len(wma_full) == 0:
            return np.full_like(arr, np.nan)
        
        raw_hma = 2 * wma_half - wma_full
        hma = wma(raw_hma, sqrt_period)
        
        # Pad to original length
        result = np.full_like(arr, np.nan)
        start_idx = period - 1
        end_idx = start_idx + len(hma)
        if end_idx <= len(arr):
            result[start_idx:end_idx] = hma
        return result
    
    hma_1w = calculate_hma(close_1w, 21)
    hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    
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
            np.isnan(hma_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Exit if price reverses and crosses the opposite Donchian band
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
        # Require volume spike (> 1.8x average)
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Long entry: Price breaks above Donchian upper band AND above weekly HMA (bullish breakout with trend confirmation)
            if (price > highest_high[i-1] and  # Breakout above previous period's high
                price > hma_1w_aligned[i]):    # Above weekly HMA (uptrend confirmation)
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short entry: Price breaks below Donchian lower band AND below weekly HMA (bearish breakdown with trend confirmation)
            elif (price < lowest_low[i-1] and    # Breakout below previous period's low
                  price < hma_1w_aligned[i]):    # Below weekly HMA (downtrend confirmation)
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals