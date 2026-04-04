#!/usr/bin/env python3
"""
Experiment #6297: 4h Donchian(20) breakout + 1d/1w pivot direction + volume confirmation
HYPOTHESIS: Tight Donchian breakouts on 4h with daily/weekly pivot filter and volume confirmation 
capture institutional momentum with proper structure. Daily pivot provides intraday bias, 
weekly pivot provides structural bias that works in both bull (breakouts above pivot in uptrend) 
and bear (breakdowns below pivot in downtrend) markets. Volume filter ensures breakouts have 
participation. Uses discrete sizing (0.25) to minimize fee churn. Target: 75-200 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_6297_4h_donchian20_1d1w_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # Precompute session hours once (open_time is already datetime64[ms])
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    # === HTF: 1d data for daily pivot (using prior day's OHLC) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 1:
        # Daily pivot = (daily_high + daily_low + daily_close) / 3
        daily_high = df_1d['high'].values
        daily_low = df_1d['low'].values  
        daily_close = df_1d['close'].values
        daily_pivot = (daily_high + daily_low + daily_close) / 3.0
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    else:
        daily_pivot_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot (using prior week's OHLC) ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 1:
        # Weekly pivot = (weekly_high + weekly_low + weekly_close) / 3
        weekly_high = df_1w['high'].values
        weekly_low = df_1w['low'].values  
        weekly_close = df_1w['close'].values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    else:
        weekly_pivot_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20-period) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 4h Indicators: Volume confirmation ===
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ratio = volume / np.where(avg_volume > 0, avg_volume, 1)
    
    # === 4h Indicators: ATR(14) for trailing stop ===
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size (discrete level)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(20, 20, 14) + 1  # Donchian, volume avg, ATR + 1
    
    for i in range(warmup, n):
        # --- Session Filter: Avoid low liquidity periods (22:00-23:59 UTC) ---
        hour = hours[i]
        if 22 <= hour <= 23:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(daily_pivot_aligned[i]) or np.isnan(weekly_pivot_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            if position_side > 0:  # Long position
                highest_since_entry = max(highest_since_entry, high[i])
                stop_price = highest_since_entry - 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks below Donchian low (failed breakout)
                # 3. Price crosses below daily pivot (intraday reversal)
                # 4. Price crosses below weekly pivot (structural reversal)
                if price <= stop_price or price <= donchian_low[i] or price < daily_pivot_aligned[i] or price < weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short position
                lowest_since_entry = min(lowest_since_entry, low[i])
                stop_price = lowest_since_entry + 2.5 * atr[i]
                # Exit conditions:
                # 1. Stoploss
                # 2. Price breaks above Donchian high (failed breakout)
                # 3. Price crosses above daily pivot (intraday reversal)
                # 4. Price crosses above weekly pivot (structural reversal)
                if price >= stop_price or price >= donchian_high[i] or price > daily_pivot_aligned[i] or price > weekly_pivot_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        breakout_up = price > donchian_high[i-1]
        breakout_down = price < donchian_low[i-1]
        volume_confirmed = volume_ratio[i] > 2.0  # Strong volume filter
        
        # Entry logic: Donchian breakout with volume AND aligned with BOTH pivots
        # LONG: breakout above Donchian high + volume + price > daily pivot AND price > weekly pivot
        # SHORT: breakout below Donchian low + volume + price < daily pivot AND price < weekly pivot
        long_entry = breakout_up and volume_confirmed and price > daily_pivot_aligned[i] and price > weekly_pivot_aligned[i]
        short_entry = breakout_down and volume_confirmed and price < daily_pivot_aligned[i] and price < weekly_pivot_aligned[i]
        
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