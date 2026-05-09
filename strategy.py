#!/usr/bin/env python3
# 12h_KAMA_Trend_With_RSI_Filter
# Hypothesis: KAMA (Kaufman Adaptive Moving Average) on 12h for trend direction, combined with RSI(14) on 12h for overbought/oversold conditions and volume confirmation on 12h. This strategy aims to capture trend continuation after pullbacks in the trend direction, avoiding choppy markets. Works in both bull and bear: KAMA adapts to market noise, RSI avoids extremes, volume confirms institutional interest.
# Uses 12h timeframe as primary, with 1w as HTF for trend confirmation (optional, but kept simple here to avoid overcomplication). Focus on high-probability entries with low trade frequency.

name = "12h_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 12h
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:  # Need enough data for KAMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # KAMA parameters: ER lookback = 10, Fast SC = 2/(2+1), Slow SC = 2/(30+1)
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_12h))
    # For first element, change is 0
    change = np.insert(change, 0, 0)
    # Sum of absolute changes over er_period
    sum_change = np.zeros_like(close_12h)
    for i in range(er_period, len(close_12h)):
        sum_change[i] = np.sum(change[i-er_period+1:i+1])
    # Absolute net change over er_period
    abs_net_change = np.abs(np.diff(close_12h, k=er_period))
    abs_net_change = np.insert(abs_net_change, [0]*er_period, 0)  # pad beginning
    # Avoid division by zero
    er = np.zeros_like(close_12h)
    valid = sum_change != 0
    er[valid] = abs_net_change[valid] / sum_change[valid]
    # Smoothing Constant (SC)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # KAMA calculation
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]  # start with first price
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align KAMA to 12h timeframe (same as primary, so no alignment needed, but using align_htf_to_ltf for safety with potential gaps)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate RSI(14) on 12h
    rsi_period = 14
    delta = np.diff(close_12h)
    delta = np.insert(delta, 0, 0)  # pad beginning
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # First average gain/loss
    avg_gain = np.zeros_like(close_12h)
    avg_loss = np.zeros_like(close_12h)
    if len(close_12h) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[1:rsi_period+1])
        avg_loss[rsi_period-1] = np.mean(loss[1:rsi_period+1])
        for i in range(rsi_period, len(close_12h)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    # Avoid division by zero
    rs = np.zeros_like(close_12h)
    valid = avg_loss != 0
    rs[valid] = avg_gain[valid] / avg_loss[valid]
    rsi = np.zeros_like(close_12h)
    rsi = 100 - (100 / (1 + rs))
    # For first rsi_period, RSI is not defined, set to 50 (neutral)
    rsi[:rsi_period] = 50
    
    # Align RSI to 12h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_12h, rsi)
    
    # Volume spike filter: current volume / 20-period average volume on 12h
    vol_ma = np.full_like(close_12h, np.nan)
    if len(close_12h) >= 20:
        vol_ma[19] = np.mean(close_12h[0:20])  # This is incorrect; should be volume mean
        # Fix: calculate volume moving average
        vol_ma = np.full_like(volume, np.nan)  # Reinitialize for volume
        if len(volume) >= 20:
            vol_ma[19] = np.mean(volume[0:20])
            for i in range(20, len(volume)):
                vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full_like(volume, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    # Align volume ratio to 12h timeframe
    volume_ratio_aligned = align_htf_to_ltf(prices, df_12h, volume_ratio)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure KAMA, RSI, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(volume_ratio_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA (uptrend), RSI < 40 (not overbought), volume spike
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 40 and 
                volume_ratio_aligned[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA (downtrend), RSI > 60 (not oversold), volume spike
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 60 and 
                  volume_ratio_aligned[i] > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought)
            if close[i] < kama_aligned[i] or rsi_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold)
            if close[i] > kama_aligned[i] or rsi_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals