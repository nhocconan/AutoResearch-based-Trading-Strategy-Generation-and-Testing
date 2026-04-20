#!/usr/bin/env python3
"""
4h_CCI_14_Trend_Volume_Filter
Hypothesis: Use 4-hour CCI(14) for mean-reversion entries in overbought/oversold zones, 
filtered by daily trend direction and volume confirmation. Enter long when CCI crosses above -100 
with daily uptrend and volume spike; enter short when CCI crosses below +100 with daily downtrend 
and volume spike. Exit when CCI crosses zero or trend reversals occur.
Designed to work in both bull and bear markets by aligning with higher timeframe trend while 
exploiting short-term mean reversion. Volume filter ensures trades occur during active participation.
Target: 20-40 trades/year with position size 0.25 to minimize fee drag.
"""

name = "4h_CCI_14_Trend_Volume_Filter"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data ONCE before loop
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 30:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_daily = df_daily['close'].values
    ema50_daily = np.full_like(close_daily, np.nan)
    if len(close_daily) >= 50:
        multiplier = 2.0 / (50 + 1)
        ema50_daily[49] = np.mean(close_daily[:50])
        for i in range(50, len(close_daily)):
            ema50_daily[i] = multiplier * close_daily[i] + (1 - multiplier) * ema50_daily[i-1]
    ema50_daily_aligned = align_htf_to_ltf(prices, df_daily, ema50_daily)
    
    # Calculate CCI(14)
    tp = (high + low + close) / 3.0  # Typical Price
    sma_tp = np.full(n, np.nan)
    if len(tp) >= 14:
        sma_tp[13] = np.mean(tp[:14])
        for i in range(14, n):
            sma_tp[i] = sma_tp[i-1] + (tp[i] - tp[i-14]) / 14.0
    
    # Mean deviation
    md = np.full(n, np.nan)
    if len(tp) >= 14:
        for i in range(14, n):
            md[i] = np.mean(np.abs(tp[i-14:i] - sma_tp[i-1]))
    
    cci = np.full(n, np.nan)
    valid = ~np.isnan(sma_tp) & ~np.isnan(md) & (md != 0)
    cci[valid] = (tp[valid] - sma_tp[valid]) / (0.015 * md[valid])
    
    # Volume spike detector: volume > 1.5 * 20-period average
    vol_ma = np.full(n, np.nan)
    if len(volume) >= 20:
        vol_ma[19] = np.mean(volume[:20])
        for i in range(20, n):
            vol_ma[i] = vol_ma[i-1] + (volume[i] - volume[i-20]) / 20.0
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(cci[i]) or np.isnan(ema50_daily_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: CCI crosses above -100 (from below) + daily uptrend + volume spike
            if (cci[i] > -100 and cci[i-1] <= -100 and 
                close[i] > ema50_daily_aligned[i] and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: CCI crosses below +100 (from above) + daily downtrend + volume spike
            elif (cci[i] < 100 and cci[i-1] >= 100 and 
                  close[i] < ema50_daily_aligned[i] and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: CCI crosses below zero OR daily trend turns down
            if (cci[i] < 0 and cci[i-1] >= 0) or close[i] < ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: CCI crosses above zero OR daily trend turns up
            if (cci[i] > 0 and cci[i-1] <= 0) or close[i] > ema50_daily_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals