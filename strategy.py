#!/usr/bin/env python3
"""
Experiment #304: 1d Donchian(20) Breakout + Weekly HMA Trend + Volume Confirmation

HYPOTHESIS: Daily Donchian breakouts aligned with weekly HMA(21) trend capture strong momentum
while avoiding counter-trend whipsaws. Weekly HMA provides smooth higher-timeframe direction
filter. Volume confirmation (2.0x average) ensures institutional participation. Designed for
1d timeframe to target 7-25 trades/year (30-100 over 4 years). Works in both bull and bear
markets by only taking breakouts in direction of weekly HMA slope.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_1d_donchian_weekly_hma_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly HMA trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly HMA(21)
    if len(df_1w) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2
        sqrt_len = int(np.sqrt(21))
        
        # WMA helper
        def wma(values, window):
            if len(values) < window:
                return np.full(len(values), np.nan)
            weights = np.arange(1, window + 1)
            return np.convolve(values, weights / weights.sum(), mode='valid')
        
        close_1w = df_1w['close'].values
        wma_half = wma(close_1w, half_len)
        wma_full = wma(close_1w, 21)
        wma_diff = 2 * wma_half - wma_full
        # Pad to original length
        wma_diff_padded = np.concatenate([np.full(half_len, np.nan), wma_diff])
        hma_21 = wma(wma_diff_padded, sqrt_len)
        # Pad to original length
        hma_21_padded = np.concatenate([np.full(sqrt_len - 1, np.nan), hma_21])
        
        # Align to 1d timeframe
        hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21_padded)
        # Weekly trend: 1 if HMA rising, -1 if falling
        hma_slope = np.diff(hma_21_aligned, prepend=hma_21_aligned[0])
        weekly_trend = np.where(hma_slope > 0, 1, -1)
    else:
        hma_21_aligned = np.full(n, np.nan)
        weekly_trend = np.full(n, 0)
    
    # === 1d Indicators ===
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
            np.isnan(vol_ma_20[i]) or np.isnan(hma_21_aligned[i]) if i < len(hma_21_aligned) else True or
            i >= len(weekly_trend)):
            signals[i] = 0.0
            continue
        
        # --- Price Channel Breakout ---
        bullish_breakout = close[i] > dc_upper_20[i]
        bearish_breakout = close[i] < dc_lower_20[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2.0x volume spike
        
        # --- Weekly HMA Trend ---
        # Only trade breakouts that align with weekly HMA trend
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
            min_hold = (i - entry_bar) >= 2  # Minimum 2 bars hold (~2 days)
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
        # Breakout above upper Donchian with volume confirmation and weekly HMA trend bullish
        if bullish_aligned and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with volume confirmation and weekly HMA trend bearish
        elif bearish_aligned and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals