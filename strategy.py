#!/usr/bin/env python3
"""
Experiment #4373: 4h Donchian Breakout + 12h HMA Trend + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 4h aligned with 12h HMA(21) trend direction and confirmed by volume (>1.5x average) capture institutional momentum with fewer false signals. The 12h HMA acts as a higher-timeframe trend filter, reducing whipsaws in ranging markets. Works in bull via upward breakouts with uptrend bias, in bear via downward breakouts with downtrend bias. Volume confirmation filters low-conviction moves. Targets 75-200 total trades over 4 years (19-50/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4373_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(open_time).hour
    
    # === Precompute HTF: 12h HMA(21) for trend filter ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) >= 21:
        # Hull Moving Average: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        close_series = pd.Series(df_12h['close'].values)
        n_12h = len(close_series)
        half_n = n_12h // 2
        sqrt_n = int(np.sqrt(n_12h))
        
        wma_half = wma(close_series.values, half_n)
        wma_full = wma(close_series.values, n_12h)
        wma_double_half = 2 * wma_half
        
        # Align lengths
        min_len = min(len(wma_double_half), len(wma_full))
        diff = wma_double_half[-min_len:] - wma_full[-min_len:]
        hma_12h = wma(diff, sqrt_n)
        
        # Pad to match original length
        hma_padded = np.full(n_12h, np.nan)
        hma_padded[-len(hma_12h):] = hma_12h
        hma_12h_values = hma_padded
        
        # Align to LTF and shift by 1 for completed bars only
        hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h_values)
    else:
        hma_12h_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) ===
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_upper = high_series.rolling(window=20, min_periods=20).max().values
    donch_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Session Filter: 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry (trailing stop)
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry (trailing stop)
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume confirmation (> 1.5x average) to filter noise
        volume_confirm = vol_ratio[i] > 1.5
        
        # 12h HMA trend: rising = uptrend, falling = downtrend
        hma_rising = hma_12h_aligned[i] > hma_12h_aligned[i-1]
        hma_falling = hma_12h_aligned[i] < hma_12h_aligned[i-1]
        
        # Donchian breakout conditions
        breakout_up = close[i] > donch_upper[i-1]  # Close above previous upper band
        breakout_down = close[i] < donch_lower[i-1]  # Close below previous lower band
        
        # Long conditions: upward breakout + uptrend + volume
        long_entry = breakout_up and hma_rising and volume_confirm
        
        # Short conditions: downward breakout + downtrend + volume
        short_entry = breakout_down and hma_falling and volume_confirm
        
        if long_entry:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_entry:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>