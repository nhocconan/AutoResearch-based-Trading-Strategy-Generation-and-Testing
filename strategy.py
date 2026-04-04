#!/usr/bin/env python3
"""
Experiment #4158: 1d Donchian(20) breakout + 1w HMA(9) trend filter + volume confirmation + ATR trailing stop
HYPOTHESIS: 1d Donchian breakouts aligned with weekly HMA(9) trend capture strong momentum moves with weekly trend filtering. Volume confirmation filters false breakouts. ATR trailing stop manages risk. Works in both bull/bear as HMA adapts to trend. Target: 30-100 total trades over 4 years (7-25/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_4158_1d_donchian20_1w_hma_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w HMA(9) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 9:
        # Calculate HMA(9): WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = 9 // 2
        sqrt_len = int(np.sqrt(9))
        wma_half = pd.Series(df_1w['close'].values).ewm(span=half_len, adjust=False).mean().values
        wma_full = pd.Series(df_1w['close'].values).ewm(span=9, adjust=False).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1w = pd.Series(raw_hma).ewm(span=sqrt_len, adjust=False).mean().values
        hma_1w_aligned = align_htf_to_ltf(prices, df_1w, hma_1w)
    else:
        hma_1w_aligned = np.full(n, np.nan)
    
    # === 1d Indicators: Donchian Channel(20) for breakout ===
    lookback_dc = 20
    highest_high = pd.Series(high).rolling(window=lookback_dc, min_periods=lookback_dc).max().values
    lowest_low = pd.Series(low).rolling(window=lookback_dc, min_periods=lookback_dc).min().values
    
    # === 1d Indicators: Volume MA(20) for confirmation ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === 1d Indicators: ATR(14) for stoploss ===
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
    
    warmup = max(lookback_dc + 1, 20 + 5, 20 + 5, 14 + 5)  # DC lookback, vol MA buffer, HMA buffer, ATR buffer
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(atr[i]) or
            np.isnan(hma_1w_aligned[i])):
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
        # Require volume spike (> 1.8x average) to filter noise
        volume_spike = vol_ratio[i] > 1.8
        
        if volume_spike:
            # Donchian breakout logic
            breakout_up = price > highest_high[i-1]
            breakout_down = price < lowest_low[i-1]
            
            # HMA trend filter
            above_hma = price > hma_1w_aligned[i]
            below_hma = price < hma_1w_aligned[i]
            
            # Long conditions: Donchian breakout up + above 1w HMA (trend alignment)
            long_entry = breakout_up and above_hma
            
            # Short conditions: Donchian breakout down + below 1w HMA (trend alignment)
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