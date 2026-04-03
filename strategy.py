#!/usr/bin/env python3
"""
Experiment #268: 12h Donchian(20) Breakout + Weekly Trend + Volume Confirmation

HYPOTHESIS: 12h Donchian breakouts aligned with weekly HMA trend (bullish/bearish) capture 
strong momentum with reduced false signals. Weekly trend provides structural bias from higher 
timeframe, while volume confirmation (2.0x average) ensures institutional participation. 
Designed for 12h timeframe to target 12-37 trades/year (50-150 over 4 years). Works in both 
bull and bear markets by only taking breakouts in direction of weekly HMA trend, with ATR-based 
trailing stoploss to manage risk.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_12h_donchian_weekly_trade_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate HMA(21) on weekly close
    if len(df_1w) >= 21:
        # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n)
        half_len = len(df_1w) // 2
        sqrt_len = int(np.sqrt(len(df_1w)))
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights/weights.sum(), mode='valid')
        close_w = df_1w['close'].values
        wma_half = wma(close_w, half_len) if half_len >= 1 else np.full_like(close_w, np.nan)
        wma_full = wma(close_w, len(close_w)) if len(close_w) >= 1 else np.full_like(close_w, np.nan)
        raw_hma = 2 * wma_half - wma_full
        hma_21 = wma(raw_hma, sqrt_len) if sqrt_len >= 1 else np.full_like(close_w, np.nan)
        # Pad to original length
        hma_padded = np.full(len(close_w), np.nan)
        start_idx = len(close_w) - len(hma_21)
        if start_idx >= 0 and len(hma_21) > 0:
            hma_padded[start_idx:] = hma_21
        weekly_hma = hma_padded
        
        # Align to 12h timeframe
        weekly_hma_aligned = align_htf_to_ltf(prices, df_1w, weekly_hma)
        
        # Weekly trend: 1 if close > HMA (bullish), -1 if close < HMA (bearish)
        weekly_trend = np.where(close[:len(weekly_hma_aligned)] > weekly_hma_aligned, 1, -1)
    else:
        weekly_hma_aligned = np.full(n, np.nan)
        weekly_trend = np.full(n, 0)
    
    # === 12h Indicators ===
    # ATR(14) for stoploss
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # Donchian Channel(20) - shift(1) to avoid look-ahead
    dc_upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    dc_lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Volume MA(20) for confirmation
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
            np.isnan(vol_ma_20[i]) or np.isnan(weekly_hma_aligned[i]) if i < len(weekly_hma_aligned) else True or
            i >= len(weekly_trend)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Weekly Trend Alignment ---
        bullish_aligned = bullish_breakout and weekly_trend[i] > 0
        bearish_aligned = bearish_breakout and weekly_trend[i] < 0
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~36h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR weekly trend turns bearish
                    if close[i] <= dc_lower_20[i] or weekly_trend[i] < 0:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR weekly trend turns bullish
                    if close[i] >= dc_upper_20[i] or weekly_trend[i] > 0:
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
        # Breakout above upper Donchian with volume confirmation and weekly trend bullish
        if bullish_aligned and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and weekly trend bearish
        elif bearish_aligned and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals

</think>