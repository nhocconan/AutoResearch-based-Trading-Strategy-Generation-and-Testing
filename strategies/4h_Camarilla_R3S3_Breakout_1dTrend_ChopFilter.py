#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_ChopFilter
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and choppiness regime filter. 
Only trade breakouts aligned with 1d trend when market is not too choppy (CHOP > 38.2 = trending).
Uses discrete position sizing (0.25) to minimize fee drag. Target: 20-50 trades/year per symbol.
Works in bull/bear via trend filter - only long in uptrend, short in downtrend.
Chop filter prevents whipsaws in ranging markets.
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
    
    # Get 1d data for EMA34 trend filter and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate Camarilla levels from previous 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla width
    rang = high_1d - low_1d
    
    # Resistance levels
    r3 = close_1d_prev + rang * 1.1 / 4
    r4 = close_1d_prev + rang * 1.1 / 2
    
    # Support levels
    s3 = close_1d_prev - rang * 1.1 / 4
    s4 = close_1d_prev - rang * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate Choppiness Index on 1d timeframe (14-period)
    # CHOP = 100 * log10(sum(ATR1) / (n * (max(high_n) - min(low_n)))) / log10(n)
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), np.abs(low_1d[1:] - close_1d[:-1])))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index
    atr1_sum = pd.Series(tr1).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr1_sum / (14 * (max_high - min_low))) / np.log10(14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Volume spike detector (20-bar volume MA on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Trend filter: only trade in direction of 1d EMA34
        uptrend = close[i] > ema_34_1d_aligned[i]
        downtrend = close[i] < ema_34_1d_aligned[i]
        
        # Regime filter: only trade when market is trending (CHOP <= 38.2)
        trending_market = chop_aligned[i] <= 38.2
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend and trending market
            if close[i] > r3_aligned[i] and volume_spike[i] and uptrend and trending_market:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend and trending market
            elif close[i] < s3_aligned[i] and volume_spike[i] and downtrend and trending_market:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below R3 OR trend changes OR market becomes choppy
            if close[i] < r3_aligned[i] or not uptrend or not trending_market:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above S3 OR trend changes OR market becomes choppy
            if close[i] > s3_aligned[i] or not downtrend or not trending_market:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_ChopFilter"
timeframe = "4h"
leverage = 1.0