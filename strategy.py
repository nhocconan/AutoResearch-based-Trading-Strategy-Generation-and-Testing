#!/usr/bin/env python3
"""
Experiment #5155: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Spike
HYPOTHESIS: On 6h timeframe, Donchian(20) breakouts aligned with weekly pivot direction (price > weekly pivot = bullish bias, price < weekly pivot = bearish bias) capture institutional momentum. Volume > 2.0x average confirms participation. Designed for 12-37 trades/year on 6h timeframe (50-150 total over 4 years) to minimize fee drag. Weekly pivot provides structural bias that works in both bull (buy dips above pivot) and bear (sell rallies below pivot) markets. Uses discrete position sizing (0.25) to minimize fee churn.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_5155_6h_donchian20_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Precompute HTF: 1d and 1w data for pivot and trend
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    # === 1d Indicators: Weekly Pivot (using prior week's OHLC) ===
    # Calculate weekly pivot from 1d data: P = (H + L + C) / 3
    if len(df_1d) >= 5:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        close_1d = df_1d['close'].values
        
        # Weekly OHLC from daily data (group by week)
        # We'll use rolling window of 5 days to approximate weekly OHLC
        # For true weekly pivot, we need prior week's H,L,C
        high_5d = pd.Series(high_1d).rolling(window=5, min_periods=5).max().values
        low_5d = pd.Series(low_1d).rolling(window=5, min_periods=5).min().values
        close_5d = pd.Series(close_1d).rolling(window=5, min_periods=5).last().values
        
        # Weekly pivot = (weekly high + weekly low + weekly close) / 3
        weekly_pivot = (high_5d + low_5d + close_5d) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian(20) channels ===
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h Indicators: Volume confirmation (2.0x spike) ===
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
    
    warmup = max(20, 20, 14, 5)  # Donchian, Volume MA, ATR, weekly pivot warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Volume filter: confirmation (>2.0x)
        vol_confirm = vol_ratio[i] > 2.0
        
        # Donchian breakout conditions with weekly pivot bias filter
        # Long: Donchian breakout above + price > weekly pivot (bullish bias)
        # Short: Donchian breakdown below + price < weekly pivot (bearish bias)
        breakout_long = (price >= high_roll[i]) and (price > weekly_pivot_aligned[i]) and vol_confirm
        breakout_short = (price <= low_roll[i]) and (price < weekly_pivot_aligned[i]) and vol_confirm
        
        # Final entry conditions
        if breakout_long:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif breakout_short:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals