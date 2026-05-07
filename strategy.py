#!/usr/bin/env python3
"""
1d_KAMA_RSI_ChopFilter_v2
Hypothesis: KAMA identifies trend direction, RSI identifies pullback entry points, and Choppiness Index filters for trending markets. This combination avoids whipsaws in sideways markets and captures trends with proper pullback entries. Works in both bull and bear markets by following the trend direction with momentum-based entries.
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
"""
name = "1d_KAMA_RSI_ChopFilter_v2"
timeframe = "1d"
leverage = 1.0

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
    
    # KAMA (Kaufman Adaptive Moving Average) - 14 period
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if False else None  # placeholder for loop
    
    # Calculate ER and SC properly
    er = np.zeros(n)
    sc = np.zeros(n)
    kama = np.zeros(n)
    
    # Initialize
    kama[0] = close[0]
    
    for i in range(1, n):
        if i < er_period:
            er[i] = 0
            sc[i] = 0
            kama[i] = kama[i-1]
            continue
            
        # Efficiency Ratio
        price_change = np.abs(close[i] - close[i-er_period])
        price_volatility = np.sum(np.abs(np.diff(close[i-er_period:i+1])))
        if price_volatility > 0:
            er[i] = price_change / price_volatility
        else:
            er[i] = 0
            
        # Smoothing Constant
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
        
        # KAMA
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI (14-period)
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    rsi = np.zeros(n)
    
    # Initial average
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
            
            if avg_loss[i] != 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))
            else:
                rsi[i] = 100
    
    # Choppiness Index (14-period) - for weekly timeframe
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate True Range for 1w
    tr1 = np.abs(df_1w['high'].values - df_1w['low'].values)
    tr2 = np.abs(df_1w['high'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]]))
    tr3 = np.abs(df_1w['low'].values - np.concatenate([[df_1w['close'].values[0]], df_1w['close'].values[:-1]]))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    
    atr_1w = np.zeros(len(df_1w))
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.mean(tr[i-13:i+1])
    
    # Chop calculation
    chop = np.zeros(len(df_1w))
    for i in range(14, len(df_1w)):
        atr_sum = np.sum(atr_1w[i-13:i+1])
        highest_high = np.max(df_1w['high'].values[i-13:i+1])
        lowest_low = np.min(df_1w['low'].values[i-13:i+1])
        if atr_sum > 0:
            chop[i] = 100 * np.log10(highest_high - lowest_low) / np.log10(14) / np.log10(atr_sum)
        else:
            chop[i] = 50
    
    chop_aligned = align_htf_to_ltf(prices, df_1w, chop, additional_delay_bars=0)
    
    # Volume filter: current volume > 1.3 * 20-period average
    vol_avg = np.zeros(n)
    for i in range(20, n):
        vol_avg[i] = np.mean(volume[i-19:i+1])
    
    volume_filter = volume > (vol_avg * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 14)  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(chop_aligned[i]) or 
            i < 20 or np.isnan(vol_avg[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA (uptrend) + RSI pullback (30-40) + trending market (CHOP < 40)
            if close[i] > kama[i] and 30 <= rsi[i] <= 40 and chop_aligned[i] < 40 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA (downtrend) + RSI bounce (60-70) + trending market (CHOP < 40)
            elif close[i] < kama[i] and 60 <= rsi[i] <= 70 and chop_aligned[i] < 40 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: trend change or RSI extreme
            if position == 1:
                if close[i] <= kama[i] or rsi[i] >= 70:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= kama[i] or rsi[i] <= 30:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals