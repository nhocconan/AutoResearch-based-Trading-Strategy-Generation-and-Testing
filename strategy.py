#!/usr/bin/env python3
"""
Experiment #215: 6h Donchian Breakout + Weekly Pivot Direction + Volume Confirmation
HYPOTHESIS: Donchian(20) breakouts on 6h timeframe aligned with weekly pivot direction (above/below weekly pivot) and volume confirmation captures institutional breakout moves. Weekly pivot provides structural support/resistance from higher timeframe, reducing false breakouts. Works in both bull/bear markets by only taking breakouts in direction of weekly trend. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_215_6h_donchian_weekly_pivot_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot and trend filter (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot (standard floor pivot) from previous week's OHLC
    weekly_pivot = np.zeros(n)
    weekly_r1 = np.zeros(n)
    weekly_s1 = np.zeros(n)
    weekly_trend_up = np.zeros(n, dtype=bool)
    weekly_trend_down = np.zeros(n, dtype=bool)
    
    # We need previous week's OHLC, so we shift the weekly data by 1
    # But align_htf_to_ltf will handle the shifting for us
    if len(df_1w) >= 2:
        prev_weekly_high = df_1w['high'].values[:-1]  # t-1 week
        prev_weekly_low = df_1w['low'].values[:-1]
        prev_weekly_close = df_1w['close'].values[:-1]
        prev_weekly_open = df_1w['open'].values[:-1]
        
        # Calculate pivot for each week (based on previous week)
        weekly_pivot_vals = (prev_weekly_high + prev_weekly_low + prev_weekly_close) / 3.0
        weekly_r1_vals = 2 * weekly_pivot_vals - prev_weekly_low
        weekly_s1_vals = 2 * weekly_pivot_vals - prev_weekly_high
        
        # Weekly trend: higher highs and higher lows = uptrend
        weekly_trend_up_vals = (prev_weekly_high > np.roll(prev_weekly_high, 1)) & (prev_weekly_low > np.roll(prev_weekly_low, 1))
        weekly_trend_down_vals = (prev_weekly_high < np.roll(prev_weekly_high, 1)) & (prev_weekly_low < np.roll(prev_weekly_low, 1))
        
        # Handle first week (no previous week)
        weekly_pivot_vals = np.concatenate([[prev_weekly_close[0]], weekly_pivot_vals])
        weekly_r1_vals = np.concatenate([[prev_weekly_close[0]], weekly_r1_vals])
        weekly_s1_vals = np.concatenate([[prev_weekly_close[0]], weekly_s1_vals])
        weekly_trend_up_vals = np.concatenate([[False], weekly_trend_up_vals])
        weekly_trend_down_vals = np.concatenate([[False], weekly_trend_down_vals])
    else:
        # Not enough data, use current values
        weekly_pivot_vals = close[:len(df_1w)]
        weekly_r1_vals = close[:len(df_1w)]
        weekly_s1_vals = close[:len(df_1w)]
        weekly_trend_up_vals = np.zeros(len(df_1w), dtype=bool)
        weekly_trend_down_vals = np.zeros(len(df_1w), dtype=bool)
    
    # Align weekly data to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot_vals)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1_vals)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1_vals)
    weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up_vals.astype(np.float64))
    weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down_vals.astype(np.float64))
    
    # === HTF: 1d data for additional regime filter (optional) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    trend_up_1d = close_1d > ema50_1d
    trend_down_1d = close_1d < ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    # === 6h Indicators: ATR(14) for stoploss ===
    tr_6h = np.zeros(n)
    tr_6h[0] = high[0] - low[0]
    for i in range(1, n):
        tr_6h[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    atr_14 = pd.Series(tr_6h).ewm(span=14, min_periods=14, adjust=False).mean().values
    
    # === 6h Indicators: Volume MA(20) for spike detection ===
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.zeros(n)
    vol_ratio[20:] = volume[20:] / vol_ma_20[20:]
    vol_ratio[:20] = 1.0
    
    # === 6h Indicators: Donchian Channel(20) ===
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # Enough for Donchian(20) and weekly data
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or
            np.isnan(weekly_s1_aligned[i]) or np.isnan(trend_up_1d_aligned[i]) or
            np.isnan(trend_down_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.8x average) ---
        volume_spike = vol_ratio[i] > 1.8
        
        # --- Weekly Pivot Position ---
        above_weekly_pivot = price > weekly_pivot_aligned[i]
        below_weekly_pivot = price < weekly_pivot_aligned[i]
        above_weekly_r1 = price > weekly_r1_aligned[i]
        below_weekly_s1 = price < weekly_s1_aligned[i]
        
        # --- Exit Logic (ATR-based stoploss) ---
        if in_position:
            bars_since_entry += 1
            
            if position_side > 0:  # Long position
                stop_level = entry_price - 2.5 * atr_14[i]  # Slightly wider stop for 6h
                if low[i] < stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian high or weekly R1 with volume
                if price >= donchian_high[i] and volume_spike:
                    # Continue if breaks above weekly R1
                    if price > weekly_r1_aligned[i]:
                        signals[i] = SIZE  # Continue trend
                    else:
                        # Take profit
                        in_position = False
                        position_side = 0
                        bars_since_entry = 0
                        signals[i] = 0.0
                        continue
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14[i]
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    bars_since_entry = 0
                    signals[i] = 0.0
                    continue
                # Take profit at Donchian low or weekly S1 with volume
                if price <= donchian_low[i] and volume_spike:
                    # Continue if breaks below weekly S1
                    if price < weekly_s1_aligned[i]:
                        signals[i] = -SIZE  # Continue trend
                    else:
                        # Take profit
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
        # Donchian breakout with volume spike and weekly pivot alignment
        # Long: Price breaks above Donchian high with volume, above weekly pivot, in uptrend
        if (price > donchian_high[i-1] and  # Break above previous Donchian high
            volume_spike and
            above_weekly_pivot and
            weekly_trend_up_aligned[i] and
            trend_up_1d_aligned[i]):
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        # Short: Price breaks below Donchian low with volume, below weekly pivot, in downtrend
        elif (price < donchian_low[i-1] and  # Break below previous Donchian low
              volume_spike and
              below_weekly_pivot and
              weekly_trend_down_aligned[i] and
              trend_down_1d_aligned[i]):
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals