#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Use weekly price action to define trend direction and daily Camarilla levels for entries.
Long only in weekly uptrend when price breaks above daily R1 with volume confirmation.
Short only in weekly downtrend when price breaks below daily S1 with volume confirmation.
Exit on close back below/above daily pivot point. Weekly trend filter reduces whipsaws in
choppy markets. Designed for 1d timeframe to keep trade frequency low (target: 10-25/year)
and minimize fee drag. Works in bull markets by buying dips/trend continuations and in bear
markets by selling rallies/trend continuations.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for trend determination
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Weekly EMA34 for trend (using Wilder's smoothing like ADX)
    close_1w = df_1w['close'].values
    ema_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 34:
        ema_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_1w[i] = (close_1w[i] * 2/35) + (ema_1w[i-1] * 33/35)
    
    # Load daily data once for Camarilla levels
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
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Align to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(ema_1w_aligned[i])):
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
        
        # Weekly trend: price above/below EMA34
        weekly_uptrend = price > ema_1w_aligned[i]
        weekly_downtrend = price < ema_1w_aligned[i]
        
        if position == 0:
            # Long conditions: weekly uptrend + break above R1 + volume
            if weekly_uptrend and price > r1_aligned[i] and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short conditions: weekly downtrend + break below S1 + volume
            elif weekly_downtrend and price < s1_aligned[i] and volume_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "1d"
leverage = 1.0