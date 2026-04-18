#!/usr/bin/env python3
"""
12h_1d_Volume_Weighted_Mean_Reversion_With_Regime_Filter
Hypothesis: Mean reversion from 1-day volume-weighted average price (VWAP) with 1-day ATR filter and 1-week trend filter.
Designed for 12h timeframe to capture multi-day mean reversion moves while avoiding false signals in strong trends.
Works in both bull and bear markets by fading extreme deviations from VWAP only when 1-week trend is weak (ADX < 25).
Target: 15-25 trades/year (~60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for VWAP and ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily VWAP (typical price * volume / cumulative volume)
    typical_price_1d = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    vwap_1d = (typical_price_1d * df_1d['volume']).cumsum() / df_1d['volume'].cumsum()
    vwap_1d_values = vwap_1d.values
    
    # Calculate daily ATR(14)
    tr1 = np.maximum(df_1d['high'], df_1d['close'].shift(1)) - np.minimum(df_1d['low'], df_1d['close'].shift(1))
    tr2 = np.abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = np.abs(df_1d['low'] - df_1d['close'].shift(1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Get weekly data for ADX trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate weekly ADX(14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr_w1 = np.maximum(high_1w[1:] - low_1w[1:], np.maximum(np.abs(high_1w[1:] - close_1w[:-1]), np.abs(low_1w[1:] - close_1w[:-1])))
    tr_w1 = np.concatenate([[np.nan], tr_w1])
    
    # Directional Movement
    dm_plus = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    dm_minus = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    dm_plus = np.concatenate([[np.nan], dm_plus])
    dm_minus = np.concatenate([[np.nan], dm_minus])
    
    # Smoothed values
    atr_w = pd.Series(tr_w1).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = np.where(atr_w != 0, 100 * dm_plus_smooth / atr_w, 0)
    di_minus = np.where(atr_w != 0, 100 * dm_minus_smooth / atr_w, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx_1w = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align all 1d indicators to 12h timeframe
    vwap_aligned = align_htf_to_ltf(prices, df_1d, vwap_1d_values)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Align weekly ADX to 12h timeframe (with 1-week delay for confirmation)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx_1w, additional_delay_bars=1)
    
    # Calculate deviation from VWAP in ATR units
    deviation = (close - vwap_aligned) / atr_aligned
    
    # Entry conditions: extreme deviation from VWAP only in weak trend (ADX < 25)
    oversold = deviation < -2.0  # Price more than 2 ATR below VWAP
    overbought = deviation > 2.0  # Price more than 2 ATR above VWAP
    weak_trend = adx_aligned < 25  # Weak trend regime
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after sufficient data for indicators
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(vwap_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(adx_aligned[i]) or atr_aligned[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for mean reversion signals in weak trend
            if oversold[i] and weak_trend[i]:
                signals[i] = 0.25
                position = 1
            elif overbought[i] and weak_trend[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: price returns to VWAP or trend strengthens
            if close[i] >= vwap_aligned[i] or adx_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price returns to VWAP or trend strengthens
            if close[i] <= vwap_aligned[i] or adx_aligned[i] >= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_Volume_Weighted_Mean_Reversion_With_Regime_Filter"
timeframe = "12h"
leverage = 1.0