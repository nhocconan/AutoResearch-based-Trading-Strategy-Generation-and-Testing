#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + ADX Regime Filter
# Bull Power = High - EMA13(close), Bear Power = EMA13(close) - Low
# Long when Bull Power > 0 AND Bear Power < 0 AND ADX(14) > 25 (trending)
# Short when Bull Power < 0 AND Bear Power > 0 AND ADX(14) > 25 (trending)
# Exit when Bull Power and Bear Power converge (both near zero) or ADX < 20
# Uses 1d EMA13 and ADX for regime alignment, 6h for entry timing
# Target: 60-120 total trades over 4 years (15-30/year) with discrete sizing 0.25
# Works in both bull (trend continuation) and bear (trend continuation) markets

name = "6h_ElderRay_ADXRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data ONCE before loop for EMA13 and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # Need enough for EMA13 and ADX calculations
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA13
    close_1d_series = pd.Series(close_1d)
    ema13_1d = close_1d_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate 1d ADX (Average Directional Index)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    up_move[0] = down_move[0] = np.nan
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_dm_14 = pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    minus_dm_14 = pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_14 / tr_14
    minus_di = 100 * minus_dm_14 / tr_14
    
    # ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    dx = np.where(np.isnan(dx) | (plus_di + minus_di) == 0, 0, dx)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Align 1d indicators to 6h
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema13_1d_aligned[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Calculate Elder Ray components for current 6h bar
        bull_power = high[i] - ema13_1d_aligned[i]
        bear_power = ema13_1d_aligned[i] - low[i]
        
        # Regime conditions
        adx_val = adx_aligned[i]
        trending = adx_val > 25
        weak_trend = adx_val < 20
        
        if position == 0:
            # Long: Strong bull power, weak bear power, trending market
            if bull_power > 0 and bear_power < 0 and trending:
                signals[i] = 0.25
                position = 1
            # Short: Strong bear power, weak bull power, trending market
            elif bear_power > 0 and bull_power < 0 and trending:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Power convergence or trend weakening
            if bull_power <= 0 or bear_power >= 0 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Power convergence or trend weakening
            if bear_power <= 0 or bull_power >= 0 or weak_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals