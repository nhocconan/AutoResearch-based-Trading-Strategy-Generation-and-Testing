#!/usr/bin/env python3
"""
Experiment #1973: 4h Donchian(20) breakout + 12h HMA trend + volume confirmation + ATR stoploss
HYPOTHESIS: Donchian(20) breakouts capture institutional order flow while 12h HMA(21) filters trend direction. 
Volume confirmation (>1.5x 20-period average) ensures breakout validity. ATR-based stoploss limits drawdown.
Works in bull/bear markets by following HTF trend with precise LTf entries. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1973_4h_donchian20_12h_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for HMA trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate 12h HMA(21)
    def hma(arr, period):
        half = period // 2
        sqrt = int(np.sqrt(period))
        wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
        wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
        raw = 2 * wma2 - wma1
        return raw.ewm(span=sqrt, adjust=False).mean().values
    
    hma_12h = hma(close_12h, 21)
    hma_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_12h)
    
    # === 4h Indicators: Donchian(20), Volume MA(20), ATR(14) ===
    # Donchian channels
    highest_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # ATR for stoploss
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0], low[0], close[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 50  # sufficient for Donchian(20), ATR(14), volume MA
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_20[i]) or np.isnan(lowest_20[i]) or
            np.isnan(hma_12h_aligned[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Update highest/lowest since entry for trailing stop
            if position_side > 0:  # Long
                highest_since_entry = max(highest_since_entry, high[i])
                # Exit if price drops 2.5*ATR from highest since entry
                if price < highest_since_entry - 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                else:
                    signals[i] = SIZE
            else:  # Short
                lowest_since_entry = min(lowest_since_entry, low[i])
                # Exit if price rises 2.5*ATR from lowest since entry
                if price > lowest_since_entry + 2.5 * atr[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require 12h HMA trend alignment for bias filter
        trend_bias = 1 if close_12h[-1] > hma_12h[-1] else -1 if len(close_12h) > 0 else 0  # Current 12h trend
        # Use aligned HMA for historical bars
        if i < len(hma_12h_aligned):
            hma_val = hma_12h_aligned[i]
            close_12h_current = close_12h[-1] if len(close_12h) > 0 else 0
            # For historical alignment, compare 12h close to its HMA at that time
            # Since we don't have 12h close aligned, we use price vs HMA as trend proxy
            trend_bias = 1 if price > hma_val else -1
        
        # Volume confirmation: require volume spike (> 1.5x average)
        volume_spike = vol_ratio[i] > 1.5
        
        if volume_spike:
            # Long entry: price breaks above Donchian upper AND 12h trend up (price > HMA)
            if trend_bias > 0 and price > highest_20[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = SIZE
            # Short entry: price breaks below Donchian lower AND 12h trend down (price < HMA)
            elif trend_bias < 0 and price < lowest_20[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                bars_since_entry = 0
                highest_since_entry = high[i]
                lowest_since_entry = low[i]
                signals[i] = -SIZE
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0
    
    return signals