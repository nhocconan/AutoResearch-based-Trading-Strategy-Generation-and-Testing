#!/usr/bin/env python3
"""
Experiment #4355: 6h Donchian Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: Donchian(20) breakouts on 6h timeframe, when aligned with weekly pivot trend (price above/below weekly pivot) and confirmed by volume spikes (>2.0x average), capture institutional momentum in both bull and bear markets. Weekly pivot provides structural bias, Donchian breakouts capture momentum, volume filters false breakouts. Targets 75-150 total trades over 4 years (19-37/year) with position size 0.25.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4355_6h_donchian20_1w_pivot_vol_v1"
timeframe = "6h"
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
    
    # === Precompute HTF: 1d Weekly Pivot (using 1d data) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Calculate weekly pivot from prior week's OHLC
        # For each 1d bar, we need the prior week's (Monday-Friday) H/L/C
        # Simplified: use rolling window of 5 days (1 week) to get weekly H/L/C
        # Then calculate pivot = (H + L + C) / 3
        # We'll align this to 6h timeframe
        
        # Get weekly high, low, close from 1d data (using 5-day rolling window)
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().shift(1)  # Prior week
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().shift(1)
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).last().shift(1)
        
        # Weekly pivot = (H + L + C) / 3
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot.values)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    # Donchian Upper = max(high, lookback=20)
    # Donchian Lower = min(low, lookback=20)
    lookback = 20
    donchian_upper = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_lower = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === 6h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback, 20, 14)  # Donchian, vol MA, ATR
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or 
            np.isnan(weekly_pivot_aligned[i])):
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
        
        # Weekly pivot bias: price above pivot = bullish bias, below = bearish bias
        price_above_pivot = price > weekly_pivot_aligned[i]
        price_below_pivot = price < weekly_pivot_aligned[i]
        
        # Donchian breakout conditions
        breakout_up = price > donchian_upper[i-1]  # Break above prior period's upper band
        breakout_down = price < donchian_lower[i-1]  # Break below prior period's lower band
        
        # Long conditions: Donchian breakout up + price above weekly pivot + volume
        long_entry = breakout_up and price_above_pivot and volume_confirm
        
        # Short conditions: Donchian breakout down + price below weekly pivot + volume
        short_entry = breakout_down and price_below_pivot and volume_confirm
        
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