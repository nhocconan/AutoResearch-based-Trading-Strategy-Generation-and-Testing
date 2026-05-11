#!/usr/bin/env python3
"""
1d_Weekly_Pivot_VWAP_Bounce
Hypothesis: On daily timeframe, price tends to bounce off weekly pivot points (R1/S1) when approaching with VWAP confirmation and low volatility (VIX-like filter). In trending markets (ADX>25), breakouts are traded; in ranging markets (ADX<20), mean reversion at pivot levels is favored. Uses weekly pivot points from 1w timeframe, VWAP from 1d, and ADX from 1d for regime filtering. Designed to work in both bull and bear markets by adapting to regime.
"""

name = "1d_Weekly_Pivot_VWAP_Bounce"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1w data for weekly pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Get 1d data for VWAP and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d OHLCV
    close_1d = prices['close'].values
    high_1d = prices['high'].values
    low_1d = prices['low'].values
    volume_1d = prices['volume'].values
    
    # --- VWAP calculation (typical price * volume) / cumulative volume ---
    typical_price = (high_1d + low_1d + close_1d) / 3.0
    vwap_numerator = np.cumsum(typical_price * volume_1d)
    vwap_denominator = np.cumsum(volume_1d)
    vwap_1d = vwap_numerator / (vwap_denominator + 1e-10)
    
    # --- ADX calculation (14 period) ---
    # True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).sum().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).sum().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # --- Weekly Pivot Points (using previous week's OHLC) ---
    prev_week_high = np.roll(df_1w['high'].values, 1)
    prev_week_low = np.roll(df_1w['low'].values, 1)
    prev_week_close = np.roll(df_1w['close'].values, 1)
    prev_week_high[0] = df_1w['high'].values[0]
    prev_week_low[0] = df_1w['low'].values[0]
    prev_week_close[0] = df_1w['close'].values[0]
    
    pivot = (prev_week_high + prev_week_low + prev_week_close) / 3.0
    r1 = 2 * pivot - prev_week_low
    s1 = 2 * pivot - prev_week_high
    r2 = pivot + (prev_week_high - prev_week_low)
    s2 = pivot - (prev_week_high - prev_week_low)
    
    # Align weekly levels to 1d
    pivot_1d = align_htf_to_ltf(prices, df_1w, pivot)
    r1_1d = align_htf_to_ltf(prices, df_1w, r1)
    s1_1d = align_htf_to_ltf(prices, df_1w, s1)
    r2_1d = align_htf_to_ltf(prices, df_1w, r2)
    s2_1d = align_htf_to_ltf(prices, df_1w, s2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start after warmup period
    start_idx = 30  # for ADX and VWAP stability
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(adx[i]) or np.isnan(vwap_1d[i]) or 
            np.isnan(pivot_1d[i]) or np.isnan(r1_1d[i]) or np.isnan(s1_1d[i])):
            if position != 0:
                # Simple stoploss: 2x ATR from entry
                atr_est = np.abs(high_1d[i] - low_1d[i])
                if position == 1 and close_1d[i] <= entry_price - 2 * atr_est:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_1d[i] >= entry_price + 2 * atr_est:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        # Regime filter: ADX > 25 = trending, ADX < 20 = ranging
        trending = adx[i] > 25
        ranging = adx[i] < 20
        
        # VWAP deviation: price close to VWAP (within 0.5%)
        vwap_dev = np.abs(close_1d[i] - vwap_1d[i]) / vwap_1d[i]
        near_vwap = vwap_dev < 0.005
        
        if position == 0:
            if trending:
                # In trending market: breakout of R1/S1 with VWAP confirmation
                if close_1d[i] > r1_1d[i] and close_1d[i] > vwap_1d[i]:
                    signals[i] = 0.25  # long breakout above R1
                    position = 1
                    entry_price = close_1d[i]
                elif close_1d[i] < s1_1d[i] and close_1d[i] < vwap_1d[i]:
                    signals[i] = -0.25  # short breakdown below S1
                    position = -1
                    entry_price = close_1d[i]
            elif ranging:
                # In ranging market: mean reversion at pivot levels with VWAP confirmation
                if i > 0:
                    # Rejection at R1 (failed breakout above) with VWAP support
                    if close_1d[i-1] > r1_1d[i-1] and close_1d[i] < r1_1d[i] and close_1d[i] > vwap_1d[i]:
                        signals[i] = -0.25  # short rejection at R1
                        position = -1
                        entry_price = close_1d[i]
                    # Rejection at S1 (failed breakdown below) with VWAP resistance
                    elif close_1d[i-1] < s1_1d[i-1] and close_1d[i] > s1_1d[i] and close_1d[i] < vwap_1d[i]:
                        signals[i] = 0.25   # long rejection at S1
                        position = 1
                        entry_price = close_1d[i]
        else:
            # Manage existing position
            if position == 1:
                # Long position management
                if trending:
                    # In trend: trail with VWAP or stop at S1
                    if close_1d[i] < vwap_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_1d[i] < s1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
                else:  # ranging or neutral
                    # In range: take profit at R2 or stop at S1
                    if close_1d[i] >= r2_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close below S1
                    elif close_1d[i] < s1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = 0.25
            elif position == -1:
                # Short position management
                if trending:
                    # In trend: trail with VWAP or stop at R1
                    if close_1d[i] > vwap_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_1d[i] > r1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
                else:  # ranging or neutral
                    # In range: take profit at S2 or stop at R1
                    if close_1d[i] <= s2_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    # Stoploss: close above R1
                    elif close_1d[i] > r1_1d[i]:
                        signals[i] = 0.0
                        position = 0
                    else:
                        signals[i] = -0.25
    
    return signals