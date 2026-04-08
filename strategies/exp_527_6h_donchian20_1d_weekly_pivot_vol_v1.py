#!/usr/bin/env python3
"""
Experiment #527: 6h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian breakouts aligned with weekly pivot bias (from 1d HTF) and volume spikes capture strong momentum with lower trade frequency. Weekly pivot provides structural support/resistance that works in both bull and bear markets by filtering breakouts against the weekly bias. Volume confirmation (>1.5x average) ensures participation. ATR-based stoploss (2.0) manages risk. Discrete position sizing (0.25) limits drawdown. Targets 75-200 total trades over 4 years by using tight entry conditions (breakout + pivot bias + volume).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_527_6h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate weekly pivot points from prior week's OHLC
    # We need to group daily data into weeks, but for simplicity we'll use rolling weekly
    # Actually, we'll calculate pivot from prior 5 days (1 week) on daily timeframe
    # Then align to 6h timeframe
    lookback_days = 5  # 1 trading week
    if len(high_1d) >= lookback_days:
        # Rolling window of prior week (excluding current day)
        highest_high_1w = pd.Series(high_1d).rolling(window=lookback_days, min_periods=lookback_days).max().shift(1).values
        lowest_low_1w = pd.Series(low_1d).rolling(window=lookback_days, min_periods=lookback_days).min().shift(1).values
        close_prev_1w = pd.Series(close_1d).rolling(window=lookback_days, min_periods=lookback_days).mean().shift(1).values
        
        # Weekly pivot: (H + L + C) / 3
        weekly_pivot = (highest_high_1w + lowest_low_1w + close_prev_1w) / 3.0
        # Weekly R1: 2*P - L
        weekly_r1 = 2 * weekly_pivot - lowest_low_1w
        # Weekly S1: 2*P - H
        weekly_s1 = 2 * weekly_pivot - highest_high_1w
        # Weekly R2: P + (H - L)
        weekly_r2 = weekly_pivot + (highest_high_1w - lowest_low_1w)
        # Weekly S2: P - (H - L)
        weekly_s2 = weekly_pivot - (highest_high_1w - lowest_low_1w)
        
        # Align to 6h timeframe
        weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
        weekly_r1_aligned = align_htf_to_ltf(prices, df_1d, weekly_r1)
        weekly_s1_aligned = align_htf_to_ltf(prices, df_1d, weekly_s1)
        weekly_r2_aligned = align_htf_to_ltf(prices, df_1d, weekly_r2)
        weekly_s2_aligned = align_htf_to_ltf(prices, df_1d, weekly_s2)
    else:
        # Not enough data
        weekly_pivot_aligned = np.full(n, np.nan)
        weekly_r1_aligned = np.full(n, np.nan)
        weekly_s1_aligned = np.full(n, np.nan)
        weekly_r2_aligned = np.full(n, np.nan)
        weekly_s2_aligned = np.full(n, np.nan)
    
    # === 6h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)  # default to 1.0 for warmup period
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 100  # sufficient for Donchian(20) warmup + other indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(weekly_pivot_aligned[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.5x average) ---
        volume_spike = vol_ratio[i] > 1.5
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Weekly Pivot Bias Filter ---
        # Bullish bias: price above weekly pivot and above S1
        bullish_bias = price > weekly_pivot_aligned[i] and price > weekly_s1_aligned[i]
        # Bearish bias: price below weekly pivot and below R1
        bearish_bias = price < weekly_pivot_aligned[i] and price < weekly_r1_aligned[i]
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.0*ATR below entry
                stop_level = entry_price - 2.0 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.0*ATR above entry
                stop_level = entry_price + 2.0 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 8 bars (~4 days on 6h) to avoid overtrading
            if bars_since_entry > 8:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + bullish weekly bias
            if breakout_up and bullish_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + bearish weekly bias
            elif breakout_down and bearish_bias:
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