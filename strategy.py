#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v1
Hypothesis: Daily KAMA trend direction + RSI mean reversion + Choppiness regime filter works in both bull and bear markets.
- KAMA (adaptive moving average) identifies trend direction with less whipsaw than standard MA
- RSI(14) < 30 for long entry, > 70 for short entry in trending markets (KAMA slope)
- Choppiness Index > 61.8 = ranging market (avoid trend signals), < 38.2 = trending market (allow trend signals)
- Volume confirmation (1.5x 20-day average) ensures institutional participation
- Weekly trend filter (close > weekly EMA34) avoids counter-trend trades in strong trends
- Designed for low frequency (target: 30-80 trades over 4 years) with discrete position sizing
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA Calculation (Adaptive Moving Average) ===
    # Efficiency Ratio = |net change| / sum of absolute changes
    # Smoothest ER = 0.1, fastest ER = 1.0
    # SC = [ER * (fastest - slowest) + slowest]^2
    # KAMA[i] = KAMA[i-1] + SC * (price[i] - KAMA[i-1])
    
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    
    # Calculate Efficiency Ratio over 10 periods
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        net_change = abs(close[i] - close[i-10])
        total_change = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if total_change > 0:
            er[i] = net_change / total_change
        else:
            er[i] = 0
    
    # Smoothing Constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # KAMA trend direction (slope)
    kama_slope = np.diff(kama, prepend=0)
    
    # === RSI Calculation ===
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # === Choppiness Index ===
    # Chop = 100 * log10(sum(ATR) / (HHV - LLB)) / log10(n)
    # Where ATR = True Range, HHV = highest high, LLB = lowest low over period
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of ATR over 14 periods
    atr_sum = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest High and Lowest Low over 14 periods
    hh = pd.Series(high).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    for i in range(14, n):
        if hh[i] - ll[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / (hh[i] - ll[i])) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # === Volume Spike ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 1.5)
    
    # === Load Weekly Data for Trend Filter ===
    df_1w = get_htf_data(prices, '1w')
    ema34_1w = pd.Series(df_1w['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # === Signals ===
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 14)  # warmup for weekly EMA, volume MA, RSI/Chop
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Regime filter: Chop > 61.8 = ranging (avoid trend signals), Chop < 38.2 = trending
        is_trending = chop[i] < 38.2
        is_ranging = chop[i] > 61.8
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema34_1w_aligned[i]
        weekly_downtrend = close[i] < ema34_1w_aligned[i]
        
        # KAMA trend
        kama_up = kama_slope[i] > 0
        kama_down = kama_slope[i] < 0
        
        # RSI signals
        rsi_oversold = rsi[i] < 30
        rsi_overbought = rsi[i] > 70
        
        if position == 0:
            # Long: KAMA up + RSI oversold + trending regime + volume spike + weekly uptrend
            if (kama_up and rsi_oversold and is_trending and 
                volume_spike[i] and weekly_uptrend):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI overbought + trending regime + volume spike + weekly downtrend
            elif (kama_down and rsi_overbought and is_trending and 
                  volume_spike[i] and weekly_downtrend):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: KAMA turns down OR RSI overbought OR weekly trend turns down
            if (not kama_up or rsi[i] > 70 or not weekly_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: KAMA turns up OR RSI oversold OR weekly trend turns up
            if (not kama_down or rsi[i] < 30 or not weekly_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_RSI_ChopFilter_v1"
timeframe = "1d"
leverage = 1.0