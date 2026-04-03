#!/usr/bin/env python3
"""
Experiment #009: 4h Donchian(20) breakout + 1d weekly pivot direction + volume confirmation
HYPOTHESIS: Price breaking 4h Donchian(20) channels with alignment to weekly pivot direction (from 1d data) captures institutional flow. Volume confirmation (>2.0x) filters false breakouts. Weekly pivot provides structural bias that works in both bull (continuation) and bear (mean reversion at extremes). Discrete sizing (0.25) controls fee drag. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_009_4h_donchian20_1d_weekly_pivot_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for weekly pivot calculation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly pivot points from prior week's OHLC
    # Weekly high = max of daily highs over prior 5 trading days (approximate)
    # Weekly low = min of daily lows over prior 5 trading days
    # Weekly close = close of 5th prior day
    # Pivot = (weekly_high + weekly_low + weekly_close) / 3
    # R1 = 2*pivot - weekly_low
    # S1 = 2*pivot - weekly_high
    # R2 = pivot + (weekly_high - weekly_low)
    # S2 = pivot - (weekly_high - weekly_low)
    
    if len(df_1d) >= 5:
        weekly_high = df_1d['high'].rolling(window=5, min_periods=5).max().shift(1).values
        weekly_low = df_1d['low'].rolling(window=5, min_periods=5).min().shift(1).values
        weekly_close = df_1d['close'].shift(5).values  # 5 days ago
        
        pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        r1 = 2 * pivot - weekly_low
        s1 = 2 * pivot - weekly_high
        r2 = pivot + (weekly_high - weekly_low)
        s2 = pivot - (weekly_high - weekly_low)
        
        # Determine bias: price above pivot = bullish bias, below = bearish bias
        # Use 1d close for bias determination
        bias_bullish = df_1d['close'].values > pivot
        bias_bearish = df_1d['close'].values < pivot
        
        # Align to 4h timeframe
        bias_bullish_aligned = align_htf_to_ltf(prices, df_1d, bias_bullish.astype(float))
        bias_bearish_aligned = align_htf_to_ltf(prices, df_1d, bias_bearish.astype(float))
        r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
        s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
        r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
        s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    else:
        # Not enough data - neutral bias
        bias_bullish_aligned = np.zeros(n)
        bias_bearish_aligned = np.zeros(n)
        r1_aligned = np.full(n, np.nan)
        s1_aligned = np.full(n, np.nan)
        r2_aligned = np.full(n, np.nan)
        s2_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 4h Indicators: Volume MA(20) for spike detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60  # sufficient for 20-period indicators + HTF warmup
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(bias_bullish_aligned[i]) or
            np.isnan(bias_bearish_aligned[i]) or np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- Exit Logic: ATR-based stoploss (using 2.5*ATR for wider stops on 4h) ---
        if in_position:
            bars_since_entry += 1
            
            # Calculate ATR(14) for stoploss
            if i >= 14:
                tr = np.zeros(i+1)
                for j in range(1, i+1):
                    tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
                tr[0] = high[0] - low[0]
                atr_val = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            else:
                atr_val = 0.0
            
            if position_side > 0:  # Long position
                # Stoploss: 2.5*ATR below entry
                stop_level = entry_price - 2.5 * atr_val
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                # Stoploss: 2.5*ATR above entry
                stop_level = entry_price + 2.5 * atr_val
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            # Optional: time-based exit after 6 bars (~1 day on 4h)
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
            # Long: breakout above upper channel AND bullish bias
            if breakout_up and bias_bullish_aligned[i] > 0.5:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND bearish bias
            elif breakout_down and bias_bearish_aligned[i] > 0.5:
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