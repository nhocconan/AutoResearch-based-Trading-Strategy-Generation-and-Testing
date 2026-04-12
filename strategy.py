#!/usr/bin/env python3
"""
4h_1d_camarilla_breakout_volume
Hypothesis: 4-hour strategy using Camarilla pivot levels from daily timeframe for breakout entries, with volume confirmation and choppiness regime filter. Works in bull/bear by only taking breakouts aligned with higher timeframe trend (12h EMA50/EMA200). Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H4 = C + 1.1 * (H-L)/2, L4 = C - 1.1 * (H-L)/2
    # Also calculate H3, L3, H2, L2, H1, L1 for reference
    camarilla_h4 = close_1d + 1.1 * range_1d / 2.0
    camarilla_l4 = close_1d - 1.1 * range_1d / 2.0
    camarilla_h3 = close_1d + 1.1 * range_1d / 4.0
    camarilla_l3 = close_1d - 1.1 * range_1d / 4.0
    camarilla_h2 = close_1d + 1.1 * range_1d / 6.0
    camarilla_l2 = close_1d - 1.1 * range_1d / 6.0
    camarilla_h1 = close_1d + 1.1 * range_1d / 12.0
    camarilla_l1 = close_1d - 1.1 * range_1d / 12.0
    
    # Daily EMA50 and EMA200 for trend direction (12h timeframe equivalent)
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Daily ATR for choppiness filter
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(np.roll(high_1d, 1) - close_1d)
    tr3 = np.abs(np.roll(low_1d, 1) - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all daily data to 4h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 4h ATR for volatility filtering
    tr_4h1 = np.abs(high - low)
    tr_4h2 = np.abs(np.roll(high, 1) - close)
    tr_4h3 = np.abs(np.roll(low, 1) - close)
    tr_4h = np.maximum(tr_4h1, np.maximum(tr_4h2, tr_4h3))
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    # Volume average for confirmation
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ema200_1d_aligned[i]) or
            np.isnan(atr_1d_aligned[i]) or np.isnan(atr_4h[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid extremely high volatility (potential whipsaw)
        atr_ratio = atr_4h[i] / atr_1d_aligned[i] if atr_1d_aligned[i] > 0 else 0
        if atr_ratio > 3.0:  # Too volatile, skip
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # Trend determination: EMA50 > EMA200 = uptrend, EMA50 < EMA200 = downtrend
        uptrend = ema50_1d_aligned[i] > ema200_1d_aligned[i]
        downtrend = ema50_1d_aligned[i] < ema200_1d_aligned[i]
        
        # Breakout conditions with volume confirmation
        if uptrend and volume_confirm and position != 1:
            # Long breakout above H4
            if close[i] > camarilla_h4_aligned[i]:
                position = 1
                signals[i] = 0.25
        elif downtrend and volume_confirm and position != -1:
            # Short breakdown below L4
            if close[i] < camarilla_l4_aligned[i]:
                position = -1
                signals[i] = -0.25
        # Exit conditions: trend reversal or opposite Camarilla level touch
        elif position == 1 and (downtrend or close[i] < camarilla_l3_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (uptrend or close[i] > camarilla_h3_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_camarilla_breakout_volume"
timeframe = "4h"
leverage = 1.0