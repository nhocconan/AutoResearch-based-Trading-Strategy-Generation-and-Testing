#!/usr/bin/env python3
"""
Experiment #190: 1d Donchian(20) Breakout + 1w HMA Trend + Volume Confirmation

HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA trend direction capture sustained momentum while avoiding false breakouts in ranging markets. Weekly HMA (21) provides smooth trend filter that works in both bull and bear markets by identifying the dominant direction. Volume confirmation ensures breakouts have institutional participation. Targets 7-25 trades/year on 1d timeframe to minimize fee drag and maximize edge.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_20_w_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on weekly close
    if len(df_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        def wma(values, window):
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights, mode='valid') / weights.sum()
        
        close_series = pd.Series(df_1w['close'])
        wma_half = close_series.rolling(window=half_len, min_periods=half_len).apply(
            lambda x: wma(x, half_len), raw=False
        ).values
        wma_full = close_series.rolling(window=21, min_periods=21).apply(
            lambda x: wma(x, 21), raw=False
        ).values
        hma_raw = 2 * wma_half - wma_full
        hma_21 = pd.Series(hma_raw).rolling(window=sqrt_len, min_periods=sqrt_len).apply(
            lambda x: wma(x, sqrt_len), raw=False
        ).values
        
        # Align to daily timeframe
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 1d Indicators ===
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_bar = -1
    highest_since_entry = 0.0
    lowest_since_entry = float('inf')
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(dc_upper_20[i]) or np.isnan(dc_lower_20[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Weekly HMA Trend ---
        # Need previous and current values to determine trend direction
        if i > 0:
            hma_now = hma_21_aligned[i]
            hma_prev = hma_21_aligned[i-1]
            hma_bullish = hma_now > hma_prev  # Rising HMA = uptrend
            hma_bearish = hma_now < hma_prev  # Falling HMA = downtrend
        else:
            hma_bullish = False
            hma_bearish = False
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 1.5 if vol_ma_20[i] > 1e-10 else False  # 1.5x volume spike
        
        # --- Position Management (Exit Logic) ---
        stop_hit = False
        
        if in_position:
            # ATR-based trailing stoploss
            if position_side > 0:
                stop_level = highest_since_entry - 2.5 * atr_14[i]
                if low[i] < stop_level:
                    stop_hit = True
            else:  # Short position
                stop_level = lowest_since_entry + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    stop_hit = True
            
            # Exit conditions: trend reversal or opposite Donchian touch
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~2 days)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR HMA turns bearish
                    if close[i] <= dc_lower_20[i] or not hma_bullish:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR HMA turns bullish
                    if close[i] >= dc_upper_20[i] or not hma_bearish:
                        stop_hit = True
            
            if stop_hit:
                signals[i] = 0.0
                in_position = False
                position_side = 0
                highest_since_entry = 0.0
                lowest_since_entry = float('inf')
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long conditions: 
        # Breakout above upper Donchian with rising weekly HMA and volume confirmation
        if bullish_breakout and hma_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with falling weekly HMA and volume confirmation
        elif bearish_breakout and hma_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals