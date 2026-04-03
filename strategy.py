#!/usr/bin/env python3
"""
Experiment #267: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian breakouts on 6h timeframe capture medium-term trends, filtered by weekly pivot direction (from 1w HTF) to ensure alignment with higher timeframe structure, and volume confirmation to avoid false breakouts. This strategy targets 12-37 trades/year (50-150 total over 4 years) by requiring confluence of three factors: price breaking Donchian channel, weekly pivot bias, and above-average volume. Works in both bull and bear markets as breakouts occur in trending environments regardless of direction.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for Donchian channel (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian(20) on 1d data
    if len(df_1d) >= 20:
        high_1d = df_1d['high'].values
        low_1d = df_1d['low'].values
        
        # Donchian upper and lower bands
        donch_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
        donch_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
        
        # Align to 6h timeframe
        donch_high_20_aligned = align_htf_to_ltf(prices, df_1d, donch_high_20)
        donch_low_20_aligned = align_htf_to_ltf(prices, df_1d, donch_low_20)
    else:
        donch_high_20_aligned = np.full(n, np.nan)
        donch_low_20_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for weekly pivot direction (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points (using prior week's OHLC)
    if len(df_1w) >= 2:
        high_1w = df_1w['high'].values
        low_1w = df_1w['low'].values
        close_1w = df_1w['close'].values
        
        # Weekly pivot: P = (H + L + C) / 3
        # Weekly bias: above P = bullish, below P = bearish
        pivot_1w = (high_1w + low_1w + close_1w) / 3.0
        weekly_bullish = close_1w > pivot_1w  # True if weekly close above pivot
        
        # Align to 6h timeframe
        weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(np.float64))
    else:
        weekly_bullish_aligned = np.full(n, np.nan)
    
    # === 6h Indicators ===
    # Volume SMA(20) for confirmation
    if n >= 20:
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    else:
        vol_sma_20 = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    
    warmup = 100  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Confirmation: Above average volume ---
        vol_confirm = volume[i] > vol_sma_20[i]
        
        # --- Breakout Conditions ---
        # Long breakout: price > Donchian upper band
        long_breakout = close[i] > donch_high_20_aligned[i]
        # Short breakout: price < Donchian lower band
        short_breakout = close[i] < donch_low_20_aligned[i]
        
        # --- Weekly Pivot Direction Filter ---
        # Only take longs in weekly bullish bias, shorts in weekly bearish bias
        weekly_bias_long = weekly_bullish_aligned[i] > 0.5
        weekly_bias_short = weekly_bullish_aligned[i] <= 0.5
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = close[entry_bar] - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian middle (mean reversion)
                donch_mid = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2.0
                if close[i] < donch_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = close[entry_bar] + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian middle
                donch_mid = (donch_high_20_aligned[i] + donch_low_20_aligned[i]) / 2.0
                if close[i] > donch_mid:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above upper band + weekly bullish bias + volume confirmation
        if long_breakout and weekly_bias_long and vol_confirm:
            in_position = True
            position_side = 1
            entry_bar = i
            signals[i] = SIZE
        # Short: Donchian breakout below lower band + weekly bearish bias + volume confirmation
        elif short_breakout and weekly_bias_short and vol_confirm:
            in_position = True
            position_side = -1
            entry_bar = i
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals