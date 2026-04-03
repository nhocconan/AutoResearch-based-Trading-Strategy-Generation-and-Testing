#!/usr/bin/env python3
"""
Experiment #107: 6h Donchian(20) breakout + weekly pivot direction + volume confirmation
HYPOTHESIS: 6h Donchian breakouts aligned with weekly Camarilla pivot direction (price above/below weekly pivot) capture institutional flow. Volume confirmation (>1.3x average) filters weak breakouts. ATR stoploss (2.0x) manages risk. Discrete position sizing (0.25) minimizes fee churn. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_107_6h_donchian20_1w_camarilla_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1d data for weekly Camarilla pivot levels (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate weekly Camarilla pivot levels from daily data (aggregated weekly)
    def resample_to_weekly(daily_prices):
        if len(daily_prices) < 5:
            return np.array([]), np.array([]), np.array([]), np.array([])
        # Simple weekly aggregation: Friday close as weekly close, weekly high/low
        weekly_high = np.maximum.reduceat(daily_prices, np.arange(0, len(daily_prices), 5))[:len(daily_prices)//5]
        weekly_low = np.minimum.reduceat(daily_prices, np.arange(0, len(daily_prices), 5))[:len(daily_prices)//5]
        weekly_close = daily_prices[4::5][:len(daily_prices)//5]
        weekly_open = daily_prices[::5][:len(daily_prices)//5]
        return weekly_high, weekly_low, weekly_close, weekly_open
    
    # For simplicity, use daily pivots as proxy for weekly direction
    # Weekly pivot = (weekly_high + weekly_low + weekly_close)/3
    # We'll approximate using 5-day rolling
    if len(df_1d) >= 5:
        weekly_high = pd.Series(df_1d['high']).rolling(window=5, min_periods=5).max().values
        weekly_low = pd.Series(df_1d['low']).rolling(window=5, min_periods=5).min().values
        weekly_close = pd.Series(df_1d['close']).rolling(window=5, min_periods=5).mean().values
        weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
        weekly_rng = weekly_high - weekly_low
        weekly_r3 = weekly_pivot + weekly_rng * 1.1 / 4
        weekly_s3 = weekly_pivot - weekly_rng * 1.1 / 4
    else:
        weekly_pivot = np.full(len(df_1d), np.nan)
        weekly_r3 = np.full(len(df_1d), np.nan)
        weekly_s3 = np.full(len(df_1d), np.nan)
    
    # Align to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1d, weekly_pivot)
    weekly_r3_aligned = align_htf_to_ltf(prices, df_1d, weekly_r3)
    weekly_s3_aligned = align_htf_to_ltf(prices, df_1d, weekly_s3)
    
    # === 6h Indicators: Donchian(20) channels ===
    donch_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 60
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donch_upper[i]) or np.isnan(donch_lower[i]) or
            np.isnan(atr_14[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r3_aligned[i]) or
            np.isnan(weekly_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Volume Confirmation: Require volume spike (> 1.3x average) ---
        volume_spike = vol_ratio[i] > 1.3
        
        # --- Donchian Breakout Conditions ---
        breakout_up = high[i] > donch_upper[i-1]
        breakout_down = low[i] < donch_lower[i-1]
        
        # --- Weekly Pivot Direction ---
        above_weekly_pivot = price > weekly_pivot_aligned[i]
        below_weekly_pivot = price < weekly_pivot_aligned[i]
        
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
                if not above_weekly_pivot and volume_spike:
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
                if not below_weekly_pivot and volume_spike:
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
        if breakout_up and volume_spike and above_weekly_pivot:
            in_position = True
            position_side = 1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = SIZE
        elif breakout_down and volume_spike and below_weekly_pivot:
            in_position = True
            position_side = -1
            entry_price = close[i]
            bars_since_entry = 0
            signals[i] = -SIZE
        else:
            signals[i] = 0.0
    
    return signals