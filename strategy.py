#!/usr/bin/env python3
"""
Experiment #059: 6h Donchian(20) Breakout + Weekly Pivot Direction + Volume Confirmation

HYPOTHESIS: Donchian breakouts on 6h capture intermediate-term trends. Weekly pivot 
direction (from 1w) filters for breakout alignment with the major trend. Volume 
confirmation (>1.5x 20-period 6h average) ensures institutional participation. 
This combination works in both bull and bear markets by only taking breakouts 
in the direction of the weekly trend, avoiding counter-trend whipsaws.
Target: 75-150 trades over 4 years (19-37/year) with discrete sizing (0.25).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "mtf_6h_donchian_weekly_pivot_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 1w data for weekly pivot trend (Call ONCE before loop) ===
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly pivot points and trend direction
    if len(df_1w) >= 2:
        # Use previous week's OHLC for pivot calculation (shifted by 1 week)
        prev_week_high = df_1w['high'].shift(1).values
        prev_week_low = df_1w['low'].shift(1).values
        prev_week_close = df_1w['close'].shift(1).values
        
        # Weekly pivot point
        weekly_pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
        
        # Trend direction: price above/below weekly pivot
        weekly_trend_up = prev_week_close > weekly_pivot  # Bullish week = look for longs
        weekly_trend_down = prev_week_close < weekly_pivot  # Bearish week = look for shorts
        
        # Align to 6h
        weekly_trend_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_up)
        weekly_trend_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_trend_down)
    else:
        weekly_trend_up_aligned = np.zeros(n, dtype=bool)
        weekly_trend_down_aligned = np.zeros(n, dtype=bool)
    
    # === HTF: 1d data for volume confirmation (Call ONCE before loop) ===
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate 6h volume ratio (current vs 20-period average) - using 1d as proxy for institutional interest
    if len(df_1d) >= 20:
        vol_1d = df_1d['volume'].values
        vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
        vol_ratio_1d = np.zeros(len(vol_1d))
        vol_ratio_1d[20:] = vol_1d[20:] / vol_ma_20[20:]
        vol_ratio_1d[:20] = 1.0  # Neutral for warmup
        vol_ratio_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ratio_1d)
    else:
        vol_ratio_1d_aligned = np.full(n, 1.0)
    
    # === 6h Donchian Channel (20-period) ===
    # Calculate Donchian high/low on 6h
    if len(high) >= 20 and len(low) >= 20:
        donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
        donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
        
        # Align Donchian levels (use previous bar's levels to avoid look-ahead)
        donchian_high_shifted = np.roll(donchian_high, 1)
        donchian_low_shifted = np.roll(donchian_low, 1)
        donchian_high_shifted[0] = donchian_high[0]  # First value
        donchian_low_shifted[0] = donchian_low[0]    # First value
    else:
        donchian_high_shifted = np.full(n, np.nan)
        donchian_low_shifted = np.full(n, np.nan)
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # Discrete position sizing (25% of capital)
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    
    warmup = 100  # Ensure enough data for indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(donchian_high_shifted[i]) or np.isnan(donchian_low_shifted[i]) or 
            np.isnan(vol_ratio_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
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
            else:  # Short position
                stop_level = entry_price + 2.5 * atr_14
                if high[i] > stop_level:
                    in_position = False
                    position_side = 0
                    signals[i] = 0.0
                    continue
            
            # Hold position
            signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic (Only if Flat) ---
        # Volume confirmation: 1d volume spike > 1.5
        volume_spike = vol_ratio_1d_aligned[i] > 1.5
        
        # Breakout logic with weekly trend filter
        if volume_spike:
            # Long breakout: price above Donchian high AND weekly trend up
            if close[i] > donchian_high_shifted[i] and weekly_trend_up_aligned[i]:
                in_position = True
                position_side = 1
                entry_price = close[i]
                signals[i] = SIZE
            # Short breakdown: price below Donchian low AND weekly trend down
            elif close[i] < donchian_low_shifted[i] and weekly_trend_down_aligned[i]:
                in_position = True
                position_side = -1
                entry_price = close[i]
                signals[i] = -SIZE
        
        # No signal
        else:
            signals[i] = 0.0
    
    return signals