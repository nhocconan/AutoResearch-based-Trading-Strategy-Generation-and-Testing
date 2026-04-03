#!/usr/bin/env python3
"""
Experiment #1959: 6h Williams %R Extreme + 12h ADX Trend Filter + Volume Spike
HYPOTHESIS: Williams %R identifies overbought/oversold conditions on 6h timeframe. 
ADX from 12h filters for trending markets (ADX > 25) to avoid false signals in chop.
Volume spike (>2x average) confirms institutional participation at extremes.
In trending markets, extremes often continue rather than reverse - we breakout in direction of trend.
Works in both bull/bear by following 12h trend direction. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_1959_6h_williamsr_12h_adx_vol_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    close = prices["close"].values.astype(np.float64)
    high = prices["high"].values.astype(np.float64)
    low = prices["low"].values.astype(np.float64)
    volume = prices["volume"].values.astype(np.float64)
    n = len(close)
    
    # === HTF: 12h data for ADX trend filter (Call ONCE before loop) ===
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ADX(14) on 12h
    period = 14
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with indices
    
    # Directional Movement
    up_move = high_12h[1:] - high_12h[:-1]
    down_move = low_12h[:-1] - low_12h[1:]
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm = np.concatenate([[0.0], plus_dm])
    minus_dm = np.concatenate([[0.0], minus_dm])
    
    # Smoothed values using Wilder's smoothing (alpha = 1/period)
    def WilderSmoothing(data, period):
        result = np.full_like(data, np.nan, dtype=np.float64)
        if len(data) >= period:
            # First value is simple average
            result[period-1] = np.nanmean(data[:period])
            # Subsequent values: Wilder smoothing
            for i in range(period, len(data)):
                if not np.isnan(result[i-1]):
                    result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    atr = WilderSmoothing(tr, period)
    plus_di = 100 * WilderSmoothing(plus_dm, period) / atr
    minus_di = 100 * WilderSmoothing(minus_dm, period) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = WilderSmoothing(dx, period)
    
    # Trend: ADX > 25 indicates trending market
    trend_strong = adx > 25.0
    trend_direction = np.where(plus_di > minus_di, 1, -1)  # +DI > -DI = uptrend
    
    # Align 12h indicators to 6h timeframe
    trend_strong_aligned = align_htf_to_ltf(prices, df_12h, trend_strong)
    trend_direction_aligned = align_htf_to_ltf(prices, df_12h, trend_direction)
    
    # === 6h Indicators: Williams %R(14) ===
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where(
        (highest_high - lowest_low) != 0,
        ((highest_high - close) / (highest_high - lowest_low)) * -100,
        -50  # neutral when no range
    )
    
    # === 6h Indicators: Volume Spike Detection ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.ones(n)
    vol_ratio[20:] = volume[20:] / vol_ma[20:]
    
    # === Signals Initialization ===
    signals = np.zeros(n)
    SIZE = 0.25  # 25% position size
    
    # Position tracking state variables
    in_position = False
    position_side = 0
    entry_price = 0.0
    bars_since_entry = 0
    
    warmup = 50  # sufficient for all indicators
    
    for i in range(warmup, n):
        # --- Data Validity Check ---
        if (np.isnan(trend_strong_aligned[i]) or np.isnan(trend_direction_aligned[i]) or
            np.isnan(williams_r[i]) or np.isnan(vol_ratio[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # --- Exit Logic ---
        if in_position:
            bars_since_entry += 1
            
            # Exit conditions
            exit_signal = False
            
            # Exit if Williams %R returns to neutral zone (-50) 
            if position_side > 0:  # Long
                if williams_r[i] >= -50:
                    exit_signal = True
            else:  # Short
                if williams_r[i] <= -50:
                    exit_signal = True
            
            # Time-based exit: max 12 bars (3 days) to prevent overstaying
            if bars_since_entry >= 12:
                exit_signal = True
            
            if exit_signal:
                in_position = False
                position_side = 0
                bars_since_entry = 0
                signals[i] = 0.0
            else:
                signals[i] = position_side * SIZE
            continue
        
        # --- New Position Entry Logic ---
        # Require strong trend on 12h
        if trend_strong_aligned[i]:
            trend_dir = trend_direction_aligned[i]
            
            # Volume confirmation: require significant spike (> 2x average)
            volume_spike = vol_ratio[i] > 2.0
            
            if volume_spike:
                # Long entry: Williams %R oversold (< -80) AND 12h uptrend
                if trend_dir > 0 and williams_r[i] < -80:
                    in_position = True
                    position_side = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = SIZE
                # Short entry: Williams %R overbought (> -20) AND 12h downtrend
                elif trend_dir < 0 and williams_r[i] > -20:
                    in_position = True
                    position_side = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                    signals[i] = -SIZE
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
        else:
            signals[i] = 0.0  # choppy market - no trades
    
    return signals