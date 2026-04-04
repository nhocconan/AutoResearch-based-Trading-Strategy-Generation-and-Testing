#!/usr/bin/env python3
"""
Experiment #4336: 12h Donchian(20) Breakout + 1d HMA Trend + Volume Spike
HYPOTHESIS: 12h Donchian breakouts capture medium-term momentum when aligned with 1d HMA(21) trend direction and confirmed by volume >2.0x average. Works in bull via upside breakouts, in bear via downside breakouts. Volume confirmation filters false breakouts. ATR(14) trailing stop (2.5x) manages risk. Targets 50-150 total trades over 4 years (12-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4336_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
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
    
    # === Precompute HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full(len(arr), np.nan)
            weights = np.arange(1, period + 1, dtype=np.float64)
            return pd.Series(arr).rolling(window=period, min_periods=period).apply(
                lambda x: np.dot(x, weights) / weights.sum(), raw=True
            ).values
        
        wma_close = wma(df_1d['close'].values, 21)
        wma_half = wma(df_1d['close'].values, half)
        wma_diff = 2 * wma_half - wma_close
        hma_1d = wma(wma_diff, sqrt_n)
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel(20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 12h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i]) or np.isnan(hma_1d_aligned[i])):
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
        # Volume confirmation: require > 2.0x average volume
        volume_confirm = vol_ratio[i] > 2.0
        
        # Trend filter: price above/below 1d HMA(21)
        price_above_hma = close[i] > hma_1d_aligned[i]
        price_below_hma = close[i] < hma_1d_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = high[i] > highest_high[i-1]  # New 20-period high
        breakout_down = low[i] < lowest_low[i-1]   # New 20-period low
        
        # Long conditions: Upside breakout + price above HMA + volume
        long_entry = breakout_up and price_above_hma and volume_confirm
        
        # Short conditions: Downside breakout + price below HMA + volume
        short_entry = breakout_down and price_below_hma and volume_confirm
        
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