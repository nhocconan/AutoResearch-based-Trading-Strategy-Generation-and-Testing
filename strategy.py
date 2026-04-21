#!/usr/bin/env python3
"""
4h_Camarilla_R1S1_Breakout_Volume_Trend
Hypothesis: Use Camarilla R1/S1 levels from 1d timeframe for mean-reversion breakouts.
In high volatility regimes (ATR > 20-period median), buy breaks above R1 with volume,
sell breaks below S1 with volume. Filter by 1d EMA34 trend to avoid counter-trend.
Designed for 4h timeframe with ~25-40 trades/year. Works in bull/bear via volatility regime.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data once for Camarilla and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels (based on previous day)
    # R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    camarilla_r1 = np.zeros_like(close_1d)
    camarilla_s1 = np.zeros_like(close_1d)
    
    for i in range(1, len(close_1d)):
        h_l = high_1d[i-1] - low_1d[i-1]
        camarilla_r1[i] = close_1d[i-1] + h_l * 1.1 / 12
        camarilla_s1[i] = close_1d[i-1] - h_l * 1.1 / 12
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # EMA34 for trend filter on 1d
    ema34 = np.zeros_like(close_1d)
    if len(close_1d) >= 34:
        ema34[33] = np.mean(close_1d[:34])
        multiplier = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34[i] = (close_1d[i] - ema34[i-1]) * multiplier + ema34[i-1]
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34)
    
    # ATR(20) for volatility regime filter
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr = np.zeros_like(tr)
    if len(tr) >= 20:
        atr[19] = np.mean(tr[:20])
        for i in range(20, len(tr)):
            atr[i] = (tr[i] * 19 + atr[i-1]) / 20
    
    # 20-period median of ATR for regime detection
    atr_median = np.full_like(atr, np.nan)
    for i in range(20, len(atr)):
        atr_median[i] = np.median(atr[i-20:i])
    atr_median_aligned = align_htf_to_ltf(prices, df_1d, atr_median)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(atr_median_aligned[i])):
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
        
        # Volatility regime: only trade when ATR > median (high vol regime)
        vol_regime = atr[i] > atr_median_aligned[i] if not np.isnan(atr_median_aligned[i]) else False
        
        if position == 0:
            # Long: break above R1 with volume in high vol regime + uptrend bias
            if (price > camarilla_r1_aligned[i] and 
                volume_ok and 
                vol_regime and 
                price > ema34_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: break below S1 with volume in high vol regime + downtrend bias
            elif (price < camarilla_s1_aligned[i] and 
                  volume_ok and 
                  vol_regime and 
                  price < ema34_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below S1 or volatility drops
            if price < camarilla_s1_aligned[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above R1 or volatility drops
            if price > camarilla_r1_aligned[i] or not vol_regime:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_Volume_Trend"
timeframe = "4h"
leverage = 1.0