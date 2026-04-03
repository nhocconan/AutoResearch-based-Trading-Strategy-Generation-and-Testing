#!/usr/bin/env python3
"""
Experiment #528: 12h Donchian(20) breakout + 1w/1d HTF bias + volume confirmation + ATR stoploss
HYPOTHESIS: 12h timeframe reduces trade frequency while Donchian breakouts capture momentum. 
HTF bias from 1w (trend) and 1d (pivot) filters breakouts to trade with higher timeframe structure. 
Volume confirmation ensures participation. ATR stoploss manages risk. Discrete sizing (0.25) controls drawdown.
Targets 50-150 total trades over 4 years by using 12h TF and tight multi-condition entry.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_528_12h_donchian20_1w_1d_bias_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for pivot bias (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot from prior day
    if len(high_1d) >= 2:
        # Prior day's OHLC
        prev_high = np.roll(high_1d, 1)
        prev_low = np.roll(low_1d, 1)
        prev_close = np.roll(close_1d, 1)
        # Set first value to nan
        prev_high[0] = np.nan
        prev_low[0] = np.nan
        prev_close[0] = np.nan
        
        # Daily pivot: (H + L + C) / 3
        daily_pivot = (prev_high + prev_low + prev_close) / 3.0
        # Daily R1: 2*P - L
        daily_r1 = 2 * daily_pivot - prev_low
        # Daily S1: 2*P - H
        daily_s1 = 2 * daily_pivot - prev_high
        
        # Align to 12h timeframe (2 bars per day)
        daily_pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
        daily_r1_aligned = align_htf_to_ltf(prices, df_1d, daily_r1)
        daily_s1_aligned = align_htf_to_ltf(prices, df_1d, daily_s1)
    else:
        daily_pivot_aligned = np.full(n, np.nan)
        daily_r1_aligned = np.full(n, np.nan)
        daily_s1_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for trend bias (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    if len(close_1w) >= 20:
        # Weekly EMA(20) for trend
        ema_1w = pd.Series(close_1w).ewm(span=20, min_periods=20, adjust=False).mean().values
        # Align to 12h timeframe (~2.5 bars per week)
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
        # Weekly trend: price > EMA20 = bullish, price < EMA20 = bearish
        weekly_trend_bull = ema_1w_aligned  # We'll compare close vs this inside loop
    else:
        ema_1w_aligned = np.full(n, np.nan)
        weekly_trend_bull = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 12h Indicators: ATR(14) for stoploss ===
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
    
    warmup = 100  # sufficient for Donchian(20) warmup + HTF alignment
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(daily_pivot_aligned[i]) or
            np.isnan(daily_r1_aligned[i]) or np.isnan(daily_s1_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HTF Bias Filters ---
        # Daily bias: price relative to daily pivot
        daily_bullish = price > daily_pivot_aligned[i]
        daily_bearish = price < daily_pivot_aligned[i]
        
        # Weekly trend bias: price relative to weekly EMA20
        weekly_bullish = price > ema_1w_aligned[i]
        weekly_bearish = price < ema_1w_aligned[i]
        
        # Combined bias: require agreement between daily and weekly
        bullish_bias = daily_bullish and weekly_bullish
        bearish_bias = daily_bearish and weekly_bearish
        
        # --- Exit Logic: ATR-based stoploss ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry (wider for 12h)
                stop_level = entry_price - 2.5 * atr[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~3 days on 12h) to avoid overtrading
            if bars_since_entry > 6:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: Donchian breakout up + bullish HTF bias (daily + weekly agreement)
            if breakout_up and bullish_bias:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: Donchian breakout down + bearish HTF bias (daily + weekly agreement)
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