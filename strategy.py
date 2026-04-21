#!/usr/bin/env python3
"""
4h_Donchian_Breakout_Volume_Trend_Filter
Hypothesis: Use Donchian(20) breakout on 4h with volume confirmation and trend filter (EMA34).
Long when price breaks above upper band with volume spike and price > EMA34.
Short when price breaks below lower band with volume spike and price < EMA34.
Exit when price crosses EMA34 or volatility collapses.
Designed for 4h timeframe to capture medium-term trends with ~20-40 trades/year.
Works in bull markets by buying breakouts and in bear markets by selling breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 4h data once for Donchian and EMA
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Donchian channels (20-period)
    lookback = 20
    upper = np.full_like(high_4h, np.nan)
    lower = np.full_like(low_4h, np.nan)
    
    for i in range(lookback, len(high_4h)):
        upper[i] = np.max(high_4h[i-lookback:i])
        lower[i] = np.min(low_4h[i-lookback:i])
    
    upper_aligned = align_htf_to_ltf(prices, df_4h, upper)
    lower_aligned = align_htf_to_ltf(prices, df_4h, lower)
    
    # EMA34 for trend filter
    ema34 = np.zeros_like(close_4h)
    if len(close_4h) >= 34:
        ema34[33] = np.mean(close_4h[:34])
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_4h)):
            ema34[i] = (close_4h[i] - ema34[i-1]) * multiplier + ema34[i-1]
    ema34_aligned = align_htf_to_ltf(prices, df_4h, ema34)
    
    # Volatility filter: ATR(14) ratio to avoid choppy markets
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    atr = np.zeros_like(tr)
    if len(tr) >= 14:
        atr[13] = np.mean(tr[:14])
        for i in range(14, len(tr)):
            atr[i] = (tr[i] * 13 + atr[i-1]) / 14
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(atr_aligned[i])):
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
        
        # Volatility filter: avoid extremely low volatility (chop)
        vol_filter = atr_aligned[i] > 0.0001  # minimum volatility threshold
        
        if position == 0:
            # Long conditions: break above upper band + volume + uptrend + volatility
            if (price > upper_aligned[i] and 
                volume_ok and 
                price > ema34_aligned[i] and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short conditions: break below lower band + volume + downtrend + volatility
            elif (price < lower_aligned[i] and 
                  volume_ok and 
                  price < ema34_aligned[i] and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or volatility collapse
            if price < ema34_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or volatility collapse
            if price > ema34_aligned[i] or not vol_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian_Breakout_Volume_Trend_Filter"
timeframe = "4h"
leverage = 1.0