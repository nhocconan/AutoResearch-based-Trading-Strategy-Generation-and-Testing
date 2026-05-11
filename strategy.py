#!/usr/bin/env python3
"""
6h_ADX_ElderRay_BullBear_v1
Hypothesis: Uses Elder Ray (Bull/Bear Power) on 6h to measure bull/bear strength, filtered by ADX trend strength.
In strong trends (ADX > 25), go long when Bull Power > 0 and short when Bear Power > 0.
In weak trends (ADX <= 25), stay flat to avoid whipsaw.
Uses 12h EMA50 as higher timeframe trend filter: only trade in direction of 12h trend.
Designed for low trade frequency (~20-30 trades/year) by requiring strong trend alignment.
Works in both bull and bear markets by adapting to trend direction.
"""

name = "6h_ADX_ElderRay_BullBear_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # --- EMA13 for Elder Ray (13-period EMA of close) ---
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # --- Elder Ray Components ---
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = ema13 - low   # Bear Power: EMA13 - Low
    
    # --- ADX (14-period) on 6h ---
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    # Directional Movement
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), 
                       np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), 
                        np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed TR, DM+
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * dm_plus14 / tr14
    minus_di = 100 * dm_minus14 / tr14
    
    # DX and ADX
    dx = np.where((plus_di + minus_di) != 0, 
                  100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # --- 12h EMA50 for trend filter ---
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if np.isnan(adx[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or np.isnan(ema50_12h_aligned[i]):
            # Maintain position if valid, otherwise flat
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filters
        strong_trend = adx[i] > 25  # Strong trend
        uptrend_12h = close[i] > ema50_12h_aligned[i]  # 12h uptrend
        downtrend_12h = close[i] < ema50_12h_aligned[i]  # 12h downtrend
        
        if position == 0:
            # Only enter in strong trends, aligned with 12h trend
            if strong_trend and uptrend_12h and bull_power[i] > 0:
                signals[i] = 0.25
                position = 1
            elif strong_trend and downtrend_12h and bear_power[i] > 0:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: trend weakens, power fades, or 12h trend changes
            if position == 1:
                exit_signal = (adx[i] <= 25) or (bull_power[i] <= 0) or (close[i] <= ema50_12h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                exit_signal = (adx[i] <= 25) or (bear_power[i] <= 0) or (close[i] >= ema50_12h_aligned[i])
                if exit_signal:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals