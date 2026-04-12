#!/usr/bin/env python3
"""
1d_1w_Camarilla_Breakout_Trend_v3
Hypothesis: On 1d timeframe, enter long when price breaks above weekly Camarilla H5 level with volume confirmation in a weekly uptrend (price > weekly EMA20), enter short when price breaks below weekly L5 level with volume confirmation in a weekly downtrend (price < weekly EMA20). Uses weekly EMA20 for trend filter and weekly Camarilla levels from prior week for structure. Volume filter ensures breakouts have institutional participation. Target: 15-25 trades per year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_Camarilla_Breakout_Trend_v3"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY INDICATORS: EMA(20) for trend filter ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # === WEEKLY INDICATOR: Prior week OHLC for Camarilla levels ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels for each week
    camarilla_H5 = close_1w + 1.1 * (high_1w - low_1w) * 1.1 / 2
    camarilla_L5 = close_1w - 1.1 * (high_1w - low_1w) * 1.1 / 2
    
    # Align to daily timeframe
    camarilla_H5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H5)
    camarilla_L5_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L5)
    
    # Volume filter: volume > 1.3 * average volume of prior 20 periods
    vol_ma = np.zeros_like(volume)
    vol_ma[20] = np.mean(volume[0:20])
    for i in range(21, len(volume)):
        vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    volume_filter = volume > 1.3 * vol_ma
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(camarilla_H5_aligned[i]) or np.isnan(camarilla_L5_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filters
        uptrend = close[i] > ema_20_1w_aligned[i]
        downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Breakout conditions with volume confirmation
        long_breakout = (close[i] > camarilla_H5_aligned[i]) and volume_filter[i]
        short_breakout = (close[i] < camarilla_L5_aligned[i]) and volume_filter[i]
        
        # Exit conditions: trend reversal or reversion to mean (close inside weekly H3/L3)
        camarilla_H3 = close_1w + 1.1 * (high_1w - low_1w) * 1.1 / 4
        camarilla_L3 = close_1w - 1.1 * (high_1w - low_1w) * 1.1 / 4
        camarilla_H3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_H3)
        camarilla_L3_aligned = align_htf_to_ltf(prices, df_1w, camarilla_L3)
        
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