#!/usr/bin/env python3
"""
6h Elder Ray + 1d Regime Filter (ADX<20 = range, ADX>25 = trend)
Hypothesis: Elder Ray (Bull Power = High - EMA13, Bear Power = EMA13 - Low) measures buying/selling pressure.
In ranging markets (ADX<20), fade extremes: short when Bull Power > 0.6*ATR, long when Bear Power > 0.6*ATR.
In trending markets (ADX>25), follow momentum: long when Bull Power > 0 and rising, short when Bear Power > 0 and rising.
Uses 1d ADX for regime to avoid whipsaws. Discrete sizing 0.25 targets ~80-120 trades over 4 years.
Works in bull/bear by adapting to 1d ADX regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ADX regime and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d ADX (14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    tr_smooth = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / tr_smooth
    di_minus = 100 * dm_minus_smooth / tr_smooth
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Calculate 1d EMA13 for Elder Ray
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_13_aligned = align_htf_to_ltf(prices, df_1d, ema_13_1d)
    
    # Calculate 6h ATR for Elder Ray scaling
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_6h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for ADX (30) + ATR (14) + EMA13 (13)
    start_idx = max(30, 14, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(adx_1d_aligned[i]) or np.isnan(ema_13_aligned[i]) or 
            np.isnan(atr_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        ema_13 = ema_13_aligned[i]
        adx = adx_1d_aligned[i]
        atr = atr_6h[i]
        
        # Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low
        bull_power = curr_high - ema_13
        bear_power = ema_13 - curr_low
        
        # Regime filters
        is_ranging = adx < 20
        is_trending = adx > 25
        
        # Exit conditions: regime change or power reversal
        if position != 0:
            exit_signal = False
            
            if position == 1:
                # Exit long if: regime turns ranging AND Bull Power fades, OR Bear Power rises
                if is_ranging and bull_power < 0.3 * atr:
                    exit_signal = True
                elif bear_power > 0.4 * atr and bear_power > bull_power:
                    exit_signal = True
                    
            elif position == -1:
                # Exit short if: regime turns ranging AND Bear Power fades, OR Bull Power rises
                if is_ranging and bear_power < 0.3 * atr:
                    exit_signal = True
                elif bull_power > 0.4 * atr and bull_power > bear_power:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
                continue
        
        # Entry conditions
        if position == 0:
            # Ranging market: fade extremes
            if is_ranging:
                # Short when Bull Power is strong (overbought)
                if bull_power > 0.6 * atr:
                    signals[i] = -0.25
                    position = -1
                # Long when Bear Power is strong (oversold)
                elif bear_power > 0.6 * atr:
                    signals[i] = 0.25
                    position = 1
            # Trending market: follow momentum
            elif is_trending:
                # Long when Bull Power positive and rising
                if bull_power > 0 and bull_power > 0.5 * (bull_power + bear_power):
                    signals[i] = 0.25
                    position = 1
                # Short when Bear Power positive and rising
                elif bear_power > 0 and bear_power > 0.5 * (bull_power + bear_power):
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            signals[i] = 0.25
        elif position == -1:
            signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_1dADX_Regime_v1"
timeframe = "6h"
leverage = 1.0