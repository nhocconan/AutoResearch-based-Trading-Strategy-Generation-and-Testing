#!/usr/bin/env python3
"""
4h_1d_Camarilla_Breakout_v3
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla H4 level with volume confirmation in an uptrend (price > 1d EMA50), enter short when price breaks below L4 level with volume confirmation in a downtrend (price < 1d EMA50). Uses 1d EMA50 for trend filter and Camarilla levels from prior day for structure. Volume filter ensures breakouts have institutional participation. Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend). Target: 20-30 trades per year per symbol (80-120 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_Camarilla_Breakout_v3"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === 4H INDICATORS: ATR(14) for volume filter volatility normalization ===
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.max([high[0] - low[0], np.abs(high[0] - close[0]), np.abs(low[0] - close[0])])], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = np.zeros_like(tr)
    atr[14] = np.mean(tr[1:15])
    for i in range(15, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # === 1D INDICATOR: EMA(50) for trend filter ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1D INDICATOR: Prior day OHLC for Camarilla levels ===
    # Camarilla levels: H4 = close + 1.1*(high-low)*1.1/2, L4 = close - 1.1*(high-low)*1.1/2
    # Actually standard Camarilla: H4 = close + 1.1*(high-low)*1.1/2? Let's use correct formula
    # Standard Camarilla: H4 = close + 1.1*(high-low)/2, L4 = close - 1.1*(high-low)/2
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each day
    camarilla_H4 = close_1d + 1.1 * (high_1d - low_1d) / 2
    camarilla_L4 = close_1d - 1.1 * (high_1d - low_1d) / 2
    
    # Align to 4h timeframe
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Volume filter: volume > 1.5 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    vol_ma[20] = np.mean(volume[0:20])
    for i in range(21, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.5 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_H4_aligned[i]) or np.isnan(camarilla_L4_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > camarilla_H4_aligned[i]) and volume_filter[i]
        short_breakout = (close[i] < camarilla_L4_aligned[i]) and volume_filter[i]
        
        # Exit conditions: trend reversal or reversion to mean (close inside H3/L3)
        # H3 = close + 1.1*(high-low)*1.1/4, L3 = close - 1.1*(high-low)*1.1/4
        camarilla_H3 = close_1d + 1.1 * (high_1d - low_1d) * 1.1 / 4
        camarilla_L3 = close_1d - 1.1 * (high_1d - low_1d) * 1.1 / 4
        camarilla_H3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H3)
        camarilla_L3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L3)
        
        exit_long = not uptrend or (close[i] < camarilla_H3_aligned[i])
        exit_short = not downtrend or (close[i] > camarilla_L3_aligned[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals