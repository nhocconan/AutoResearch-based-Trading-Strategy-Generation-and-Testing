#!/usr/bin/env python3
"""
Experiment #269: 4h Donchian(20) breakout + 1d EMA(50) trend + 1w volume spike filter

HYPOTHESIS: Combining Donchian channel breakouts on 4h for trend continuation signals with 1d EMA trend alignment and 1w volume spike confirmation creates a robust trend-following strategy that works in both bull and bear markets. The 4h Donchian(20) captures intermediate-term breakouts, the 1d EMA(50) ensures alignment with the primary trend, and the 1w volume spike filter confirms institutional participation. Targets 25-50 trades/year on 4h timeframe (100-200 total over 4 years) to minimize fee drag while capturing high-probability trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_4h_donchian_ema_volspike_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for EMA trend (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA(50) on 1d close
    if len(df_1d) >= 50:
        close_1d = df_1d['close'].values
        ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
        ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    else:
        ema_50_1d_aligned = np.full(n, np.nan)
    
    # === HTF: 1w data for volume spike filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate volume ratio: current week volume / 4-week average volume
    if len(df_1w) >= 4:
        vol_1w = df_1w['volume'].values
        vol_ma_4 = pd.Series(vol_1w).rolling(window=4, min_periods=4).mean().values
        vol_ratio_1w = np.full(len(vol_1w), np.nan)
        valid = vol_ma_4 > 0
        vol_ratio_1w[valid] = vol_1w[valid] / vol_ma_4[valid]
        
        # Align to 4h timeframe
        vol_ratio_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ratio_1w)
    else:
        vol_ratio_1w_aligned = np.full(n, np.nan)
    
    # === 4h Indicators ===
    # Donchian Channel(20) - upper and lower bands
    donchian_window = 20
    if n >= donchian_window:
        highest_high = pd.Series(high).rolling(window=donchian_window, min_periods=donchian_window).max().values
        lowest_low = pd.Series(low).rolling(window=donchian_window, min_periods=donchian_window).min().values
    else:
        highest_high = np.full(n, np.nan)
        lowest_low = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = max(100, donchian_window)  # Ensure enough data for HTF and indicator calculations
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # --- Volume Spike Filter: Require 1.5x average volume ---
        if vol_ratio_1w_aligned[i] < 1.5:
            signals[i] = 0.0
            continue
        
        # --- Donchian Breakout Signals ---
        breakout_up = high[i] > highest_high[i-1]  # Current high exceeds previous period's highest high
        breakout_down = low[i] < lowest_low[i-1]   # Current low exceeds previous period's lowest low
        
        # --- 1d EMA Trend Alignment ---
        price_above_ema = close[i] > ema_50_1d_aligned[i]
        price_below_ema = close[i] < ema_50_1d_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            # Calculate ATR(14) for stoploss
            tr = np.zeros(i+1)
            tr[0] = high[0] - low[0]
            for j in range(1, i+1):
                tr[j] = max(high[j] - low[j], abs(high[j] - close[j-1]), abs(low[j] - close[j-1]))
            atr_14 = pd.Series(tr).ewm(span=14, min_periods=14, adjust=False).mean().iloc[-1]
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if high[i] > entry_price + 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
                # Take profit at 3R (7.5 * ATR)
                if low[i] < entry_price - 7.5 * atr_14:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout up with price above 1d EMA (bullish alignment)
        if breakout_up and price_above_ema:
            in_position = True
            position_side = 1
            entry_price = close[i]
            signals[i] = SIZE
        # Short: Donchian breakout down with price below 1d EMA (bearish alignment)
        elif breakout_down and price_below_ema:
            in_position = True
            position_side = -1
            entry_price = close[i]
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals