#!/usr/bin/env python3
"""
12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_V1
Hypothesis: Breakout of Camarilla R1/S1 levels on 12h timeframe with 1d trend filter (EMA34) and volume confirmation.
Works in bull/bear: In uptrend (price > 1d EMA34), buy R1 breakout; in downtrend (price < 1d EMA34), sell S1 breakout.
Uses 1d EMA for trend, 1w close for bias filter. Target: 12-37 trades/year per symbol (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla levels and trend
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
    
    # Camarilla levels: R1, S1 (primary breakout levels)
    rang = prev_high - prev_low
    r1 = prev_close + rang * 1.0 / 12
    s1 = prev_close - rang * 1.0 / 12
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Load 1w data for weekly bias filter (long-term direction)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    prev_close_1w = np.roll(close_1w, 1)
    prev_close_1w[0] = np.nan
    weekly_bias = align_htf_to_ltf(prices, df_1w, prev_close_1w)  # weekly close as trend bias
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(weekly_bias[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 1.5 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 1.5 * vol_ma
        else:
            volume_ok = False
        
        # Determine bias: weekly close > previous weekly close = bullish bias
        weekly_bullish = weekly_bias[i] > weekly_bias[i-1] if i > 0 and not np.isnan(weekly_bias[i-1]) else True
        
        # Trend determination: price vs 1d EMA34
        uptrend = price > ema_34_1d_aligned[i]
        downtrend = price < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long conditions: price > R1 (breakout) AND uptrend AND weekly bullish bias AND volume
            if (price > r1_aligned[i] and 
                uptrend and 
                weekly_bullish and 
                volume_ok):
                signals[i] = 0.25
                position = 1
            # Short conditions: price < S1 (breakdown) AND downtrend AND weekly bearish bias AND volume
            elif (price < s1_aligned[i] and 
                  downtrend and 
                  not weekly_bullish and 
                  volume_ok):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < 1d EMA34 (trend reversal) or price < S1 (mean reversion)
            if price < ema_34_1d_aligned[i] or price < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > 1d EMA34 (trend reversal) or price > R1 (mean reversion)
            if price > ema_34_1d_aligned[i] or price > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1d_1w_Camarilla_R1S1_Breakout_Volume_Regime_V1"
timeframe = "12h"
leverage = 1.0