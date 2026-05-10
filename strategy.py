#!/usr/bin/env python3
"""
4h_Triple_EMA_Volume_Momentum
Hypothesis: EMA crossover (8/21/55) with volume momentum and ATR volatility filter.
Goes long when fast EMA crosses above medium EMA, price above slow EMA, and volume > 1.5x 20-period average.
Goes short when fast EMA crosses below medium EMA, price below slow EMA, and volume > 1.5x average.
Uses 1d EMA50 as trend filter to avoid counter-trend trades.
Designed for 4h timeframe to target 20-40 trades/year, minimizing fee drag.
Works in bull/bear by aligning with higher timeframe trend.
"""

name = "4h_Triple_EMA_Volume_Momentum"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # EMA crossovers: fast(8), medium(21), slow(55)
    ema8 = np.full(n, np.nan)
    ema21 = np.full(n, np.nan)
    ema55 = np.full(n, np.nan)
    
    if n >= 55:
        # Initialize EMAs
        ema8[7] = np.mean(close[:8])
        ema21[20] = np.mean(close[:21])
        ema55[54] = np.mean(close[:55])
        
        alpha8 = 2 / (8 + 1)
        alpha21 = 2 / (21 + 1)
        alpha55 = 2 / (55 + 1)
        
        for i in range(55, n):
            ema8[i] = alpha8 * close[i] + (1 - alpha8) * ema8[i-1]
            ema21[i] = alpha21 * close[i] + (1 - alpha21) * ema21[i-1]
            ema55[i] = alpha55 * close[i] + (1 - alpha55) * ema55[i-1]
    
    # Volume filter: 20-period average
    vol_ma20 = np.full(n, np.nan)
    if n >= 20:
        vol_ma20[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma20[i] = (vol_ma20[i-1] * 19 + volume[i]) / 20
    
    # ATR for volatility stop (14-period)
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr[13] = np.mean(tr[:14])
        for i in range(14, n):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 55  # Wait for EMA55
    
    for i in range(start_idx, n):
        if np.isnan(ema8[i]) or np.isnan(ema21[i]) or np.isnan(ema55[i]) or np.isnan(vol_ma20[i]) or np.isnan(atr[i]) or np.isnan(ema50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirm = volume[i] > 1.5 * vol_ma20[i]
        
        # EMA crossover conditions
        ema8_above_21 = ema8[i] > ema21[i]
        ema8_below_21 = ema8[i] < ema21[i]
        price_above_ema55 = close[i] > ema55[i]
        price_below_ema55 = close[i] < ema55[i]
        
        # Trend filter from 1d EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: EMA8 crosses above EMA21, price above EMA55, uptrend, volume
            if ema8_above_21 and price_above_ema55 and is_uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: EMA8 crosses below EMA21, price below EMA55, downtrend, volume
            elif ema8_below_21 and price_below_ema55 and is_downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: EMA8 crosses below EMA21 or price closes below EMA55 or trend turns down
            if ema8_below_21 or not price_above_ema55 or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: EMA8 crosses above EMA21 or price closes above EMA55 or trend turns up
            if ema8_above_21 or not price_below_ema55 or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals