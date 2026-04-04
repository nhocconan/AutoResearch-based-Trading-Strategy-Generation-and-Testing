#!/usr/bin/env python3
"""
Experiment #4350: 1d Donchian(20) + 1w HMA(21) + Volume Spike
HYPOTHESIS: Daily Donchian breakouts capture medium-term trends, filtered by weekly HMA trend direction and confirmed by volume spikes (>2.0x 20-day average). Works in bull via upside breakouts above rising weekly HMA, in bear via downside breakouts below falling weekly HMA. Targets 30-100 total trades over 4 years (7-25/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4350_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
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
    
    # === Precompute HTF: 1w HMA(21) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 21:
        # Calculate Weighted Moving Average (WMA) for HMA
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # HMA = WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_double = 2 * wma_half
        
        # Align arrays by trimming from start
        diff = wma_double[:len(wma_full)] - wma_full
        hma_1w = wma(diff, sqrt_len)
        
        # Pad beginning with NaN to match original length
        hma_1w_full = np.full(len(close_1w), np.nan)
        hma_1w_full[len(close_1w) - len(hma_1w):] = hma_1w
        
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w_full)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian(20) channels ===
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1w_aligned[i])):
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
        # Require volume confirmation (> 2.0x average) to filter noise
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter: price relative to weekly HMA
        price_above_hma = price > hma_1w_aligned[i]
        price_below_hma = price < hma_1w_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = price > highest_20[i]
        breakout_down = price < lowest_20[i]
        
        if volume_confirm:
            # Long conditions: upside breakout + price above weekly HMA
            long_entry = breakout_up and price_above_hma
            
            # Short conditions: downside breakout + price below weekly HMA
            short_entry = breakout_down and price_below_hma
            
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
        else:
            signals[i] = 0.0
    
    return signals