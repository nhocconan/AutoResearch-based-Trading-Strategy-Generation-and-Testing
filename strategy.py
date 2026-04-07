#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: On 12h timeframe, use Camarilla pivot levels from daily timeframe for entry/exit, filtered by weekly trend using Supertrend and volume confirmation. Camarilla levels provide natural support/resistance, weekly Supertrend ensures we trade with higher timeframe trend, and volume confirmation reduces false breaks. Works in both bull/bear markets by following weekly trend and using mean-reversion at Camarilla levels during ranging periods. Target: 50-150 trades over 4 years (12-37/year).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Parameters
    atr_period = 14
    vol_lookback = 20
    vol_threshold = 1.5
    
    # Calculate ATR for volatility
    high = prices['high'].values
    low = prices['low'].values
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    # Calculate volume moving average
    vol_ma = pd.Series(volume).ewm(span=vol_lookback, adjust=False, min_periods=vol_lookback).mean().values
    
    # Load daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from daily OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: H4, L4, H3, L3, H2, L2, H1, L1
    # H4 = Close + 1.5 * (High - Low)
    # L4 = Close - 1.5 * (High - Low)
    # H3 = Close + 1.25 * (High - Low)
    # L3 = Close - 1.25 * (High - Low)
    # H2 = Close + 1.166 * (High - Low)
    # L2 = Close - 1.166 * (High - Low)
    # H1 = Close + 1.0833 * (High - Low)
    # L1 = Close - 1.0833 * (High - Low)
    
    camarilla_h4 = close_1d + 1.5 * (high_1d - low_1d)
    camarilla_l4 = close_1d - 1.5 * (high_1d - low_1d)
    camarilla_h3 = close_1d + 1.25 * (high_1d - low_1d)
    camarilla_l3 = close_1d - 1.25 * (high_1d - low_1d)
    camarilla_h2 = close_1d + 1.166 * (high_1d - low_1d)
    camarilla_l2 = close_1d - 1.166 * (high_1d - low_1d)
    camarilla_h1 = close_1d + 1.0833 * (high_1d - low_1d)
    camarilla_l1 = close_1d - 1.0833 * (high_1d - low_1d)
    
    # Align Camarilla levels to 12h timeframe
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    h2_12h = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    l2_12h = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    h1_12h = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    l1_12h = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    
    # Load weekly Supertrend for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < atr_period:
        return np.zeros(n)
    
    # Calculate weekly Supertrend
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    tr1_1w = high_1w[1:] - low_1w[1:]
    tr2_1w = np.abs(high_1w[1:] - close_1w[:-1])
    tr3_1w = np.abs(low_1w[1:] - close_1w[:-1])
    tr_1w = np.concatenate([[np.max([high_1w[0] - low_1w[0], np.abs(high_1w[0] - close_1w[0]), np.abs(low_1w[0] - close_1w[0])])], 
                            np.maximum(tr1_1w, np.maximum(tr2_1w, tr3_1w))])
    atr_1w = pd.Series(tr_1w).ewm(alpha=1/atr_period, adjust=False, min_periods=atr_period).mean().values
    
    hl2_1w = (high_1w + low_1w) / 2
    upper_band_1w = hl2_1w + (3.0 * atr_1w)
    lower_band_1w = hl2_1w - (3.0 * atr_1w)
    
    supertrend_1w = np.full(len(df_1w), np.nan)
    direction_1w = np.full(len(df_1w), 1)
    
    for i in range(atr_period, len(df_1w)):
        if np.isnan(atr_1w[i]) or np.isnan(upper_band_1w[i]) or np.isnan(lower_band_1w[i]):
            continue
            
        if i == atr_period:
            supertrend_1w[i] = upper_band_1w[i]
            direction_1w[i] = -1
        else:
            if close_1w[i] <= supertrend_1w[i-1]:
                supertrend_1w[i] = upper_band_1w[i]
                direction_1w[i] = -1
            else:
                supertrend_1w[i] = lower_band_1w[i]
                direction_1w[i] = 1
            
            if direction_1w[i] == 1:
                if lower_band_1w[i] < lower_band_1w[i-1]:
                    lower_band_1w[i] = lower_band_1w[i-1]
            else:
                if upper_band_1w[i] > upper_band_1w[i-1]:
                    upper_band_1w[i] = upper_band_1w[i-1]
            
            if direction_1w[i] == 1:
                supertrend_1w[i] = lower_band_1w[i]
            else:
                supertrend_1w[i] = upper_band_1w[i]
    
    supertrend_1w_aligned = align_htf_to_ltf(prices, df_1w, supertrend_1w)
    direction_1w_aligned = align_htf_to_ltf(prices, df_1w, direction_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(max(atr_period, vol_lookback), n):
        if (np.isnan(close[i]) or np.isnan(h4_12h[i]) or np.isnan(l4_12h[i]) or 
            np.isnan(h3_12h[i]) or np.isnan(l3_12h[i]) or np.isnan(h2_12h[i]) or 
            np.isnan(l2_12h[i]) or np.isnan(h1_12h[i]) or np.isnan(l1_12h[i]) or
            np.isnan(supertrend_1w_aligned[i]) or np.isnan(direction_1w_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        volume_ok = vol_ratio > vol_threshold
        
        weekly_uptrend = direction_1w_aligned[i] == 1
        
        if position == 1:  # Long position
            # Exit: price closes below L3 (strong support broken)
            if close[i] < l3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above H3 (strong resistance broken)
            if close[i] > h3_12h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if volume_ok:
                # Long setup: price tests L4 or L3 in uptrend
                if weekly_uptrend:
                    if l4_12h[i] <= close[i] <= l3_12h[i]:
                        position = 1
                        signals[i] = 0.25
                else:
                    # Short setup: price tests H4 or H3 in downtrend
                    if h3_12h[i] <= close[i] <= h4_12h[i]:
                        position = -1
                        signals[i] = -0.25
    
    return signals