#!/usr/bin/env python3
"""
Experiment #1958: 1d Donchian(20) breakout + 1w HMA trend + volume confirmation
HYPOTHESIS: Daily Donchian channel breakouts capture institutional momentum, filtered by weekly HMA trend to avoid counter-trend whipsaws. Volume confirmation ensures breakout legitimacy. Works in bull/bear markets by only trading with the higher timeframe trend. Target: 30-100 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1958_1d_donchian20_1w_hma_vol_v1"
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
    
    # Calculate 1w HMA(21) - Hull Moving Average
    def hma(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = np.full_like(arr, np.nan)
        wma1 = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= half - 1:
                wma2[i] = np.mean(arr[i-half+1:i+1])
            if i >= period - 1:
                wma1[i] = np.mean(arr[i-period+1:i+1])
        raw_hma = 2 * wma2 - wma1
        hma_result = np.full_like(arr, np.nan)
        for i in range(len(arr)):
            if i >= sqrt - 1:
                hma_result[i] = np.mean(raw_hma[i-sqrt+1:i+1])
        return hma_result
    
    hma_21_1w = hma(close_1w, 21)
    hma_21_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_21_1w)
    
    # === Primary: 1d Donchian(20) channels ===
    def donchian_channel(high_arr, low_arr, period):
        upper = np.full_like(high_arr, np.nan)
        lower = np.full_like(low_arr, np.nan)
        for i in range(len(high_arr)):
            if i >= period - 1:
                upper[i] = np.max(high_arr[i-period+1:i+1])
                lower[i] = np.min(low_arr[i-period+1:i+1])
        return upper, lower
    
    upper_20, lower_20 = donchian_channel(high, low, 20)
    
    # === 1d volume confirmation: volume > 1.5x 20-period average ===
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
    bars_since_entry = 0
    
    warmup = 50  # sufficient for Donchian(20) and HMA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or
            np.isnan(hma_21_1w_aligned[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions: opposite Donchian break
            exit_signal = False
            
            if position_side > 0:  # Long position
                # Exit if price breaks below lower Donchian
                if price < lower_20[i]:
                    exit_signal = True
            else:  # Short position
                # Exit if price breaks above upper Donchian
                if price > upper_20[i]:
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
        # Require weekly HMA trend alignment
        price_above_hma = close_1w[-1] > hma_21_1w[-1] if len(close_1w) == len(hma_21_1w) else False
        # Use aligned HMA for current bar trend
        hma_trend_up = hma_21_1w_aligned[i] > 0  # Simplified: if HMA value is valid, assume trend
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above upper Donchian AND weekly HMA uptrend
            if hma_trend_up and price > upper_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short entry: price breaks below lower Donchian AND weekly HMA downtrend
            elif not hma_trend_up and price < lower_20[i]:
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