#!/usr/bin/env python3
"""
1d_Camarilla_R1S1_Breakout_Volume_ChopFilter_v1
Hypothesis: Breakout of Camarilla R1/S1 levels on daily timeframe with volume confirmation and choppiness regime filter.
Works in bull/bear: In trending markets (CHOP < 38.2), breakouts continue; in ranging markets (CHOP > 61.8), fade extremes.
Uses daily Camarilla levels from previous day, volume spike confirmation, and weekly trend bias from 1w EMA200.
Target: 15-25 trades/year per symbol (60-100 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1 (breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 1d timeframe (no shift needed as we use previous day's levels)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Volume filter: current volume > 2.0 * 20-period average
    volume_ma = prices['volume'].rolling(window=20, min_periods=1).mean().values
    volume_ok = prices['volume'].values > (2.0 * volume_ma)
    
    # Choppiness regime filter on 1d timeframe
    # CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (trend follow)
    hl_range = np.maximum(high_1d, np.roll(high_1d, 1)) - np.minimum(low_1d, np.roll(low_1d, 1))
    atr_14 = pd.Series(hl_range).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    chop = 100 * np.log10(sum_atr_14 / np.log10(14)) / np.log10(hl_range)
    chop = np.where(hl_range == 0, 50, chop)  # avoid division by zero
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Weekly trend bias: 1w EMA200 slope
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_slope = np.diff(ema_200_1w, prepend=ema_200_1w[0])
    weekly_bullish = align_htf_to_ltf(prices, df_1w, ema_200_slope > 0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(chop_aligned[i]) or np.isnan(weekly_bullish[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        
        # Determine market regime
        is_trending = chop_aligned[i] < 38.2
        is_ranging = chop_aligned[i] > 61.8
        
        if position == 0:
            # Long breakout: price > R1 AND volume confirmation AND (trending OR weekly bullish bias in ranging)
            if (price > r1_aligned[i] and 
                volume_ok[i] and 
                (is_trending or (is_ranging and weekly_bullish[i]))):
                signals[i] = 0.25
                position = 1
            # Short breakout: price < S1 AND volume confirmation AND (trending OR weekly bearish bias in ranging)
            elif (price < s1_aligned[i] and 
                  volume_ok[i] and 
                  (is_trending or (is_ranging and not weekly_bullish[i]))):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < S1 (mean reversion) or volatility expansion
            if price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > R1 (mean reversion) or volatility expansion
            if price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Camarilla_R1S1_Breakout_Volume_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0