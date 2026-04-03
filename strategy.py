#!/usr/bin/env python3
"""
Experiment #012: 12h Donchian(20) breakout + 1d HMA trend + volume confirmation
HYPOTHESIS: Price breaking 12h Donchian(20) channels with alignment to daily HMA trend captures institutional flow. Volume confirmation (>2.0x) filters false breakouts. Daily HMA provides structural bias that works in both bull (continuation) and bear (mean reversion at extremes). Discrete sizing (0.25) controls fee drag. Target: 50-150 total trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_012_12h_donchian20_1d_hma_vol_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    open_time = prices["open_time"].values
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on daily close
    if len(df_1d) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = df_1d['close'].rolling(window=10, min_periods=10).mean()
        full_len = df_1d['close'].rolling(window=21, min_periods=21).mean()
        raw_hma = 2 * half_len - full_len
        hma_21 = raw_hma.rolling(window=int(np.sqrt(21)), min_periods=int(np.sqrt(21))).mean()
        hma_values = hma_21.values
        hma_aligned = align_htf_to_ltf(prices, df_1d, hma_values)
        # Trend: price above HMA = bullish, below = bearish
        hma_trend_aligned = hma_aligned  # Will compare close vs hma_aligned inside loop
    else:
        hma_trend_aligned = np.full(n, np.nan)
    
    # === 12h Indicators: Donchian Channel (20) ===
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # === 12h Indicators: Volume MA(20) for spike detection ===
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
            np.isnan(vol_ratio[i]) or np.isnan(hma_trend_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        hma_val = hma_trend_aligned[i]
        
        # --- Volume Confirmation: Require volume spike (> 2.0x average) ---
        volume_spike = vol_ratio[i] > 2.0
        
        # --- Donchian Breakout Conditions ---
        breakout_up = price > highest_high[i]
        breakout_down = price < lowest_low[i]
        
        # --- HMA Trend Conditions ---
        price_above_hma = price > hma_val
        price_below_hma = price < hma_val
        
        # --- Exit Logic: ATR-based stoploss (using 2.5*ATR for wider stops) ---
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
            
            # Optional: time-based exit after 10 bars (~10 days on 12h)
            if bars_since_entry > 10:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        if volume_spike:
            # Long: breakout above upper channel AND price above HMA (bullish trend)
            if breakout_up and price_above_hma:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                signals[i] = SIZE
            # Short: breakout below lower channel AND price below HMA (bearish trend)
            elif breakout_down and price_below_hma:
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