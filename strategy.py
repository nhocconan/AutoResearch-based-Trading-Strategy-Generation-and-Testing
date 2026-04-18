#!/usr/bin/env python3
"""
1d_KAMA_Trend_Filter_Volume_Spike
Hypothesis: KAMA adapts to market noise, reducing false signals in chop. 
In trending markets (price > KAMA), long on retracements to KAMA with volume spike. 
In ranging markets (price near KAMA), short deviations at Bollinger Bands with volume spike.
Works in both bull (trend following) and bear (mean reversion at extremes) markets.
Target: 10-25 trades/year on 1d timeframe with disciplined entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_ktf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-week data for trend filter (higher timeframe)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate KAMA on daily close
    def kama(price, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(price, n=length))
        volatility = np.sum(np.abs(np.diff(price, n=1)), axis=0)
        er = np.zeros_like(price)
        er[length:] = change / (volatility + 1e-10)
        # Smoothing constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        # KAMA
        kama = np.zeros_like(price)
        kama[:] = np.nan
        kama[length] = price[length]
        for i in range(length+1, len(price)):
            kama[i] = kama[i-1] + sc[i] * (price[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close, length=10, fast=2, slow=30)
    
    # Bollinger Bands (20, 2)
    bb_length = 20
    bb_mult = 2
    bb_basis = np.zeros(n)
    bb_dev = np.zeros(n)
    bb_upper = np.zeros(n)
    bb_lower = np.zeros(n)
    for i in range(bb_length-1, n):
        bb_basis[i] = np.mean(close[i-bb_length+1:i+1])
        bb_dev[i] = bb_mult * np.std(close[i-bb_length+1:i+1])
        bb_upper[i] = bb_basis[i] + bb_dev[i]
        bb_lower[i] = bb_basis[i] - bb_dev[i]
    
    # Volume spike: current volume > 2.0 x 20-period average
    vol_ma = np.zeros(n)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 2.0)
    
    # 1-week EMA34 trend filter
    ema34_1w = np.zeros(len(close_1w))
    ema34_1w[:] = np.nan
    k = 2 / (34 + 1)
    for i in range(34, len(close_1w)):
        if i == 34:
            ema34_1w[i] = np.mean(close_1w[0:35])
        else:
            ema34_1w[i] = close_1w[i] * k + ema34_1w[i-1] * (1 - k)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure indicators ready
    
    for i in range(start_idx, n):
        if (np.isnan(kama_vals[i]) or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Trending market: price > weekly EMA34
            if close[i] > ema34_1w_aligned[i]:
                # Long on retracement to KAMA with volume spike
                if close[i] <= kama_vals[i] * 1.01 and close[i] >= kama_vals[i] * 0.99 and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
            else:
                # Mean reversion in ranging market: short at upper BB, long at lower BB
                if close[i] >= bb_upper[i] and vol_spike[i]:
                    signals[i] = -0.25
                    position = -1
                elif close[i] <= bb_lower[i] and vol_spike[i]:
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Long exit: price crosses above KAMA (trend) or touches upper BB (mean reversion)
            if close[i] > ema34_1w_aligned[i]:
                # Trending exit: close above KAMA
                if close[i] > kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                # Ranging exit: touch upper BB
                if close[i] >= bb_upper[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses below KAMA (trend) or touches lower BB (mean reversion)
            if close[i] > ema34_1w_aligned[i]:
                # Trending exit: close above KAMA
                if close[i] > kama_vals[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                # Ranging exit: touch lower BB
                if close[i] <= bb_lower[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_Filter_Volume_Spike"
timeframe = "1d"
leverage = 1.0