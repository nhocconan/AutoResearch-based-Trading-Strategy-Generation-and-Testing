#!/usr/bin/env python3
"""
Experiment #4338: 1d Donchian(20) breakout + 1w HMA(21) trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts capture institutional accumulation/distribution when aligned with weekly HMA trend and volume confirmation. Works in bull via upward breakouts above weekly HMA, in bear via downward breakouts below weekly HMA. Weekly trend filter prevents whipsaw in ranging markets. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4338_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === Precompute HTF: 1w HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate HMA(21): HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1, dtype=np.float64)
            return np.convolve(arr, weights / weights.sum(), mode='valid')
        
        # Pad with NaN for proper alignment
        wma_half = np.full(len(df_1w), np.nan)
        wma_full = np.full(len(df_1w), np.nan)
        
        if len(df_1w) >= half_len:
            wma_vals = wma(df_1w['close'].values, half_len)
            wma_half[half_len-1:half_len-1+len(wma_vals)] = wma_vals
        
        if len(df_1w) >= 21:
            wma_vals = wma(df_1w['close'].values, 21)
            wma_full[20:20+len(wma_vals)] = wma_vals
        
        # HMA = WMA(2*WMA(half) - WMA(full), sqrt(length))
        hma_input = 2 * wma_half - wma_full
        hma_1w = np.full(len(df_1w), np.nan)
        
        if len(df_1w) >= sqrt_len:
            hma_vals = wma(hma_input[~np.isnan(hma_input)], sqrt_len)
            # Find valid start index
            valid_start = ~np.isnan(hma_input)
            if np.any(valid_start):
                first_valid = np.where(valid_start)[0][0]
                hma_1w[first_valid + sqrt_len - 1:first_valid + sqrt_len - 1 + len(hma_vals)] = hma_vals
        
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    def rolling_max(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.max(arr[i-window+1:i+1])
        return result
    
    def rolling_min(arr, window):
        result = np.full(len(arr), np.nan)
        for i in range(window-1, len(arr)):
            result[i] = np.min(arr[i-window+1:i+1])
        return result
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i])):
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
        
        # Trend filter: price relative to weekly HMA
        price_above_weekly_hma = price > hma_1w_aligned[i]
        price_below_weekly_hma = price < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        donch_breakout_up = price > donch_high[i-1]  # Break above previous period's high
        donch_breakout_down = price < donch_low[i-1]  # Break below previous period's low
        
        # Long conditions: Upward breakout + price above weekly HMA + volume
        long_entry = donch_breakout_up and price_above_weekly_hma and volume_confirm
        
        # Short conditions: Downward breakout + price below weekly HMA + volume
        short_entry = donch_breakout_down and price_below_weekly_hma and volume_confirm
        
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