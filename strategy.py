#!/usr/bin/env python3
"""
Experiment #3991: 6h Donchian(20) breakout + 1d weekly pivot + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly pivot direction (from 1d HTF) capture sustained moves. 
Weekly pivot levels act as institutional support/resistance: price > weekly R1 = bullish bias, 
price < weekly S1 = bearish bias. Volume > 1.5x MA(20) confirms breakout strength. 
ATR(20) trailing stop (2.5x) manages risk. Discrete sizing (0.25) reduces fee drag. 
Target: 75-200 trades over 4 years (19-50/year). Works in bull/bear via weekly pivot as trend filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_3991_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation ===
    df_1d = get_htf_data(prices, '1d')
    # Calculate weekly pivot from daily OHLC (using prior week's data)
    # We'll use rolling window of 5 trading days (1 week) to approximate weekly pivot
    lookback_week = 5
    if len(df_1d) >= lookback_week:
        # Get last completed week's OHLC (shift by 1 to avoid look-ahead)
        week_high = pd.Series(df_1d['high'].values).rolling(window=lookback_week, min_periods=lookback_week).max().shift(1).values
        week_low = pd.Series(df_1d['low'].values).rolling(window=lookback_week, min_periods=lookback_week).min().shift(1).values
        week_close = pd.Series(df_1d['close'].values).rolling(window=lookback_week, min_periods=lookback_week).mean().shift(1).values
        
        # Calculate weekly pivot points (standard formula)
        pivot = (week_high + week_low + week_close) / 3.0
        r1 = 2 * pivot - week_low
        s1 = 2 * pivot - week_high
        r2 = pivot + (week_high - week_low)
        s2 = pivot - (week_high - week_low)
        r3 = week_high + 2 * (pivot - week_low)
        s3 = week_low - 2 * (week_high - pivot)
        
        # Align to LTF (6h) with proper shift(1) for completed bars only
        pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
        r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
        s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    else:
        # Not enough data for weekly pivot
        pivot_aligned = r1_aligned = s1_aligned = r2_aligned = s2_aligned = r3_aligned = s3_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(20) for volatility and trailing stop ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(span=20, min_periods=20, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = max(lookback_dc + 1, 20, 20, lookback_week + 1)  # DC lookback, vol MA, ATR, weekly data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i])):
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
        # Require volume spike (> 1.5x average) to filter noise
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Determine trend alignment from weekly pivot
            # Bullish bias: price above weekly R1
            # Bearish bias: price below weekly S1
            bullish_bias = price > r1_aligned[i]
            bearish_bias = price < s1_aligned[i]
            
            # Breakout conditions
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            if bullish_bias and breakout_up:
                in_position = True
                position_side = 1
                entry_price = close[i]
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            elif bearish_bias and breakout_down:
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