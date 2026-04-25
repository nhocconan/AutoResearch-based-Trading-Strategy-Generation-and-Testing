#!/usr/bin/env python3
"""
12h_KAMA_Trend_Regime_Volume
Hypothesis: 12h KAMA trend with choppiness regime filter and volume confirmation.
KAMA adapts to market noise, reducing whipsaw in choppy regimes. 
Choppiness Index > 61.8 = range (mean reversion), < 38.2 = trending (trend follow).
Volume > 1.5x 20-bar MA confirms breakout strength.
Works in bull/bear: regime filter adapts strategy to market conditions.
Target: 15-25 trades/year (~60-100 over 4 years) to minimize fee drag.
Discrete sizing 0.25 balances profit and fees.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for HTF filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h KAMA (adaptive trend)
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # will fix below
    
    # Proper ER calculation
    er = np.zeros(n)
    for i in range(10, n):
        change_val = np.abs(close[i] - close[i-10])
        volatility_val = np.sum(np.abs(np.diff(close[i-10:i+1])))
        if volatility_val > 0:
            er[i] = change_val / volatility_val
        else:
            er[i] = 1.0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate Choppiness Index (14-period)
    chop = np.zeros(n)
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], np.abs(high[i] - close[i-1]), np.abs(low[i] - close[i-1]))
    
    for i in range(14, n):
        atr_14[i] = np.mean(tr[i-13:i+1])
        hh = np.max(high[i-13:i+1])
        ll = np.min(low[i-13:i+1])
        if atr_14[i] > 0:
            chop[i] = 100 * np.log10(atr_14[i] * 14 / (hh - ll)) / np.log10(14)
        else:
            chop[i] = 50  # neutral
    
    # Align HTF indicators
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA is 12h but we use 1d for alignment safety
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (30), chop (14), volume MA (20)
    start_idx = max(30, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(chop_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: chop > 61.8 = range (mean revert), chop < 38.2 = trending (trend follow)
            if chop_aligned[i] < 38.2:  # Trending regime
                # Long: price > KAMA AND 1d trend bullish AND volume confirm
                long_setup = (close[i] > kama_aligned[i]) and \
                             (close[i] > ema_34_1d_aligned[i]) and \
                             volume_confirm[i]
                # Short: price < KAMA AND 1d trend bearish AND volume confirm
                short_setup = (close[i] < kama_aligned[i]) and \
                              (close[i] < ema_34_1d_aligned[i]) and \
                              volume_confirm[i]
            else:  # Range regime (chop >= 38.2)
                # Long: price < KAMA (mean reversion to upside) AND volume confirm
                long_setup = (close[i] < kama_aligned[i]) and volume_confirm[i]
                # Short: price > KAMA (mean reversion to downside) AND volume confirm
                short_setup = (close[i] > kama_aligned[i]) and volume_confirm[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: regime change to range OR price crosses KAMA in wrong direction
            if chop_aligned[i] >= 38.2 or close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: regime change to range OR price crosses KAMA in wrong direction
            if chop_aligned[i] >= 38.2 or close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_KAMA_Trend_Regime_Volume"
timeframe = "12h"
leverage = 1.0