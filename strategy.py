#!/usr/bin/env python3
"""
Experiment #3554: 1h Donchian Breakout + 4h/1d Pivot + Volume Confirmation
HYPOTHESIS: 1h Donchian(20) breakouts with 4h/1d pivot direction filter and volume confirmation capture intraday momentum while minimizing trades. 
4h pivot provides medium-term trend, 1d pivot provides institutional levels. Volume confirms breakout strength. 
Position size 0.20. Target: 80-150 total trades over 4 years (20-38/year).
Uses 4h/1d for signal direction, 1h only for entry timing and risk management.
Session filter (08-20 UTC) reduces noise trades. Works in bull (continuation from pivot support) and bear (continuation from pivot resistance) via price channels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3554_1h_donchian20_4h_1d_pivot_vol_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # Pre-compute session hours for filtering (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    
    # === HTF: 4h data for pivot and trend filter (Call ONCE before loop) ===
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # === HTF: 1d data for institutional pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 4h pivot points (using prior day's data)
    lookback_4h_day = 6  # 6 * 4h = 24h
    prior_4h_high = pd.Series(high_4h).rolling(window=lookback_4h_day, min_periods=lookback_4h_day).max().shift(1).values
    prior_4h_low = pd.Series(low_4h).rolling(window=lookback_4h_day, min_periods=lookback_4h_day).min().shift(1).values
    prior_4h_close = pd.Series(close_4h).rolling(window=lookback_4h_day, min_periods=lookback_4h_day).mean().shift(1).values
    
    # 4h pivot formula: P = (H + L + C) / 3
    pivot_4h = (prior_4h_high + prior_4h_low + prior_4h_close) / 3.0
    # Resistance 1: R1 = 2*P - L
    r1_4h = 2 * pivot_4h - prior_4h_low
    # Support 1: S1 = 2*P - H
    s1_4h = 2 * pivot_4h - prior_4h_high
    
    # Calculate 1d pivot points (using prior week's data)
    lookback_1d_week = 5  # 5 trading days
    prior_1d_high = pd.Series(high_1d).rolling(window=lookback_1d_week, min_periods=lookback_1d_week).max().shift(1).values
    prior_1d_low = pd.Series(low_1d).rolling(window=lookback_1d_week, min_periods=lookback_1d_week).min().shift(1).values
    prior_1d_close = pd.Series(close_1d).rolling(window=lookback_1d_week, min_periods=lookback_1d_week).mean().shift(1).values
    
    # 1d weekly pivot formula: P = (H + L + C) / 3
    pivot_1d = (prior_1d_high + prior_1d_low + prior_1d_close) / 3.0
    # Resistance 1: R1 = 2*P - L
    r1_1d = 2 * pivot_1d - prior_1d_low
    # Support 1: S1 = 2*P - H
    s1_1d = 2 * pivot_1d - prior_1d_high
    
    # Align all pivot levels to 1h timeframe
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    r1_4h_aligned = align_htf_to_ltf(prices, df_4h, r1_4h)
    s1_4h_aligned = align_htf_to_ltf(prices, df_4h, s1_4h)
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 1h Indicators: Donchian channels (20-period) for entry timing ===
    lookback_1h = 20
    highest_high_1h = pd.Series(high).rolling(window=lookback_1h, min_periods=lookback_1h).max().values
    lowest_low_1h = pd.Series(low).rolling(window=lookback_1h, min_periods=lookback_1h).min().values
    
    # === 1h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1h Indicators: ATR(14) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.20  # 20% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(50, lookback_1h, lookback_4h_day + 1, lookback_1d_week + 1, 20, 14)
    
    for i in range(warmup, n):
        # --- Session Filter: Only trade 08-20 UTC ---
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # --- Data Validity Check ---
        if (np.isnan(highest_high_1h[i]) or np.isnan(lowest_low_1h[i]) or
            np.isnan(pivot_4h_aligned[i]) or np.isnan(r1_4h_aligned[i]) or np.isnan(s1_4h_aligned[i]) or
            np.isnan(pivot_1d_aligned[i]) or np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR below highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks below 4h or 1d support (mean reversion)
                elif price < s1_4h_aligned[i] or price < s1_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR above lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                # Exit if price breaks above 4h or 1d resistance (mean reversion)
                elif price > r1_4h_aligned[i] or price > r1_1d_aligned[i]:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require volume spike (> 2.0x average) for confirmation
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Determine market bias relative to 4h and 1d pivot
            price_vs_4h_pivot = price - pivot_4h_aligned[i]
            price_vs_1d_pivot = price - pivot_1d_aligned[i]
            
            # Long entry: price breaks above 1h Donchian high with bullish bias (above both pivots)
            if (price > highest_high_1h[i] and 
                price_vs_4h_pivot > 0 and 
                price_vs_1d_pivot > 0):
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below 1h Donchian low with bearish bias (below both pivots)
            elif (price < lowest_low_1h[i] and 
                  price_vs_4h_pivot < 0 and 
                  price_vs_1d_pivot < 0):
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

</think>