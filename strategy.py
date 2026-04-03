#!/usr/bin/env python3
"""
Experiment #207: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian channel breakouts on 6h timeframe filtered by weekly pivot direction (from 1w HTF) and volume confirmation captures institutional breakout moves while avoiding false breakouts in ranging markets. Weekly pivot provides structural support/resistance levels that work in both bull and bear regimes by aligning with higher timeframe market structure. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_207_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate EMA50 on 1d close for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_down_1d = close_1d < ema50_1d
    
    # Align to 6h timeframe
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === HTF: 1w data for weekly pivot direction ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot point and levels from previous week
    weekly_pivot = np.zeros(n)
    weekly_r1 = np.zeros(n)
    weekly_s1 = np.zeros(n)
    
    for i in range(len(df_1w)):
        idx_start = i * 7 * 4  # Approximate: 1w = 7*4 6h bars
        idx_end = min((i + 1) * 7 * 4, n)
        if idx_start >= n:
            break
            
        # Use previous week's OHLC for current week's pivot
        if i > 0:
            prev_week_high = df_1w['high'].iloc[i-1]
            prev_week_low = df_1w['low'].iloc[i-1]
            prev_week_close = df_1w['close'].iloc[i-1]
            
            pivot_point = (prev_week_high + prev_week_low + prev_week_close) / 3.0
            weekly_pivot[idx_start:idx_end] = pivot_point
            weekly_r1[idx_start:idx_end] = pivot_point + (prev_week_high - prev_week_low)
            weekly_s1[idx_start:idx_end] = pivot_point - (prev_week_high - prev_week_low)
    
    # Forward fill any remaining values
    weekly_pivot = pd.Series(weekly_pivot).ffill().bfill().values
    weekly_r1 = pd.Series(weekly_r1).ffill().bfill().values
    weekly_s1 = pd.Series(weekly_s1).ffill().bfill().values
    
    # Weekly trend: price above/below weekly pivot
    weekly_trend_up = close > weekly_pivot
    weekly_trend_down = close < weekly_pivot
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Donchian Channel (20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or
            np.isnan(weekly_trend_up[i]) or np.isnan(weekly_trend_down[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Breakout Conditions ---
        bullish_breakout = price > donchian_high[i]
        bearish_breakout = price < donchian_low[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.0 * atr_14[i]
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches Donchian midpoint (take profit)
                if price >= donchian_mid[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            else:  # Short position
                stop_level = entry_price + 2.0 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Exit if price reaches Donchian midpoint (take profit)
                if price <= donchian_mid[i]:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
            
            if bars_since_entry < 2:
                signals[i] = position_side * SIZE
                continue
            
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Long: Donchian breakout above with 1d uptrend, weekly uptrend, and volume spike
        if (bullish_breakout and 
            trend_up_1d_aligned[i] and 
            weekly_trend_up[i] and 
            volume_spike):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Donchian breakdown below with 1d downtrend, weekly downtrend, and volume spike
        elif (bearish_breakout and 
              trend_down_1d_aligned[i] and 
              weekly_trend_down[i] and 
              volume_spike):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals