#!/usr/bin/env python3
"""
Experiment #361: 4h Donchian(20) breakout + HMA(21) trend + volume confirmation + ATR stoploss

HYPOTHESIS: Donchian channel breakouts capture strong momentum moves. Combining with 
HMA(21) trend filter ensures we trade in the direction of the intermediate trend, while 
volume confirmation (>1.5x average) filters false breakouts. ATR-based stoploss (2.5x) 
manages risk. Targets 20-50 trades/year on 4h timeframe (80-200 total over 4 years) 
for minimal fee drag. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend filter (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        close_1d = df_1d['close'].values
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half = 21 // 2
        sqrt_n = int(np.sqrt(21))
        wma_half = pd.Series(close_1d).rolling(window=half, min_periods=half).mean().values
        wma_full = pd.Series(close_1d).rolling(window=21, min_periods=21).mean().values
        raw_hma = 2 * wma_half - wma_full
        hma_1d = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
        hma_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_1d)
    else:
        hma_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for additional trend context (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate EMA(50) on 1w close for long-term trend
    if len(df_1w) >= 50:
        close_1w = df_1w['close'].values
        ema_50_1w = pd.Series(close_1w).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    else:
        ema_50_1w_aligned = np.full(n, np.nan)
    
    # === LTF: Donchian(20) channels on 4h ===
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # === LTF: Volume confirmation on 4h ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0  # Neutral for warmup
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(hma_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        # --- Exit Logic (ATR-based stoploss or time-based) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                # Update highest high since entry
                if high[i] > highest_since_entry:
                    highest_since_entry = high[i]
                # ATR stoploss
                stop_level = highest_since_entry - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Time-based exit: hold max 20 bars (~5 days on 4h)
                # Not implemented to keep simple
            else:  # Short position
                # Update lowest low since entry
                if low[i] < lowest_since_entry:
                    lowest_since_entry = low[i]
                # ATR stoploss
                stop_level = lowest_since_entry + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Trend filters: 
        #   - HMA(21) on 1d: price above = uptrend, below = downtrend
        #   - EMA(50) on 1w: price above = long-term uptrend
        is_uptrend = close[i] > hma_1d_aligned[i] and close[i] > ema_50_1w_aligned[i]
        is_downtrend = close[i] < hma_1d_aligned[i] and close[i] < ema_50_1w_aligned[i]
        
        # Volume confirmation: require >1.5x average volume
        volume_confirmed = vol_ratio[i] > 1.5
        
        # Long: Donchian breakout above upper channel in uptrend with volume
        long_condition = is_uptrend and (close[i] > highest_high[i]) and volume_confirmed
        
        # Short: Donchian breakdown below lower channel in downtrend with volume
        short_condition = is_downtrend and (close[i] < lowest_low[i]) and volume_confirmed
        
        if long_condition:
            in_position = True
            position_side = 1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = SIZE
        elif short_condition:
            in_position = True
            position_side = -1
            entry_price = close[i]
            highest_since_entry = high[i]
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>