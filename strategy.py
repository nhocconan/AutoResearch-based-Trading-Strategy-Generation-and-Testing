#!/usr/bin/env python3
"""
Experiment #4160: 4h Donchian(20) breakout + 1d HMA(21) trend filter + volume confirmation + ATR stoploss
HYPOTHESIS: 4h Donchian breakouts aligned with daily HMA(21) trend capture strong momentum with less lag than EMA. Volume confirmation (>2.0x average) filters false breakouts. ATR trailing stop (2.5x) manages risk. Works in both bull/bear as HMA adapts to trend. Target: 75-200 total trades over 4 years (19-50/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4160_4h_donchian20_1d_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d HMA(21) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) >= 21:
        # Hull Moving Average calculation
        half_len = df_1d['close'].rolling(window=21//2, min_periods=21//2).mean()
        sqrt_len = int(np.sqrt(21))
        wma_half = 2 * half_len
        wma_full = df_1d['close'].rolling(window=21, min_periods=21).mean()
        diff = wma_half - wma_full
        hma_1d = diff.rolling(window=sqrt_len, min_periods=sqrt_len).mean().values
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === 4h Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 4h Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 4h Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 5, 21 + 5, 14 + 5)  # DC lookback, vol MA buffer, HMA buffer, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1d_aligned[i])):
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
        # Require volume spike (> 2.0x average) to filter noise
        volume_spike = vol_ratio[i] > 2.0
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # HMA trend filter
            above_hma = price > hma_1d_aligned[i]
            below_hma = price < hma_1d_aligned[i]
            
            # Long conditions: Donchian breakout up + above 1d HMA (trend alignment)
            long_entry = breakout_up and above_hma
            
            # Short conditions: Donchian breakout down + below 1d HMA (trend alignment)
            short_entry = breakout_down and below_hma
            
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
        else:
            signals[i] = 0.0
    
    return signals