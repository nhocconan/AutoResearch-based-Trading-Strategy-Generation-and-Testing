#!/usr/bin/env python3
"""
Experiment #237: 4h Donchian(20) Breakout + 1d HMA Trend + Volume Spike + ATR Stoploss

HYPOTHESIS: 4h Donchian breakouts aligned with 1d HMA trend direction capture medium-term momentum with institutional participation. Volume spike confirms breakout validity. This combination works in both bull and bear markets by following established trends while filtering false breakouts. Targets 19-50 trades/year on 4h timeframe (75-200 total over 4 years) to balance opportunity with fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_hma_volume_1d_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for HMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate HMA(21) on 1d close
    if len(df_1d) >= 21:
        # Hull Moving Average: WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        half_len = 21 // 2  # 10
        sqrt_len = int(np.sqrt(21))  # 4
        
        def wma(arr, period):
            if len(arr) < period:
                return np.full_like(arr, np.nan)
            weights = np.arange(1, period + 1)
            return np.convolve(arr, weights, mode='valid') / weights.sum()
        
        close_1d = df_1d['close'].values
        wma_half = np.full_like(close_1d, np.nan)
        wma_full = np.full_like(close_1d, np.nan)
        
        for i in range(len(close_1d)):
            if i >= half_len - 1:
                wma_half[i] = wma(close_1d[max(0, i-half_len+1):i+1], half_len)[-1]
            if i >= 21 - 1:
                wma_full[i] = wma(close_1d[max(0, i-21+1):i+1], 21)[-1]
        
        raw_hma = 2 * wma_half - wma_full
        hma_21 = np.full_like(raw_hma, np.nan)
        for i in range(len(raw_hma)):
            if i >= sqrt_len - 1 and not np.isnan(raw_hma[i]):
                hma_21[i] = wma(raw_hma[max(0, i-sqrt_len+1):i+1], sqrt_len)[-1]
        
        # Align to 4h timeframe
        hma_21_aligned = align_htf_to_ltf(prices, df_1d, hma_21)
    else:
        hma_21_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
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
        
        # --- HTF Trend Filter ---
        # Use 1d HMA for trend direction
        hma_trend_bullish = close[i] > hma_21_aligned[i]
        hma_trend_bearish = close[i] < hma_21_aligned[i]
        
        # --- Volume Confirmation ---
        vol_ok = volume[i] > vol_ma_20[i] * 2.0 if vol_ma_20[i] > 1e-10 else False  # 2x volume spike
        
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
            min_hold = (i - entry_bar) >= 3  # Minimum 3 bars hold (~12h)
            if min_hold:
                if position_side > 0:
                    # Exit long: price touches lower Donchian OR breaks below 1d HMA
                    if close[i] <= dc_lower_20[i] or close[i] < hma_21_aligned[i]:
                        stop_hit = True
                else:  # position_side < 0
                    # Exit short: price touches upper Donchian OR breaks above 1d HMA
                    if close[i] >= dc_upper_20[i] or close[i] > hma_21_aligned[i]:
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
        # Breakout above upper Donchian with price above 1d HMA and volume confirmation
        if bullish_breakout and hma_trend_bullish and vol_ok:
            in_position = True
            position_side = 1
            entry_bar = i
            highest_since_entry = high[i]
            signals[i] = SIZE
        # Short conditions:
        # Breakout below lower Donchian with price below 1d HMA and volume confirmation
        elif bearish_breakout and hma_trend_bearish and vol_ok:
            in_position = True
            position_side = -1
            entry_bar = i
            lowest_since_entry = low[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals