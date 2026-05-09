#!/usr/bin/env python3
# 4h_KAMA_Direction_RSI20_80_Volume1.5
# Hypothesis: KAMA direction (trend) + RSI extremes (overbought/oversold) + volume >1.5x 20-bar average.
# KAMA adapts to market noise, reducing whipsaw in sideways markets. RSI 20/80 captures strong momentum extremes.
# Volume filter ensures only high-conviction moves trigger entries. Designed for 15-30 trades/year on 4h timeframe.
# Works in bull markets by buying oversold dips in uptrends, in bear markets by selling overbought rallies in downtrends.

name = "4h_KAMA_Direction_RSI20_80_Volume1.5"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for KAMA calculation (same timeframe as primary)
    # KAMA requires efficiency ratio and smoothing constant
    lookback = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    
    # Calculate Efficiency Ratio (ER) and smoothed KAMA
    change = np.abs(np.diff(close, n=lookback))
    volatility = np.sum(np.abs(np.diff(close)), axis=0) if lookback == 1 else \
                 np.array([np.sum(np.abs(np.diff(close[i:i+lookback]))) for i in range(len(close)-lookback+1)])
    
    # Pad arrays to match length
    change_padded = np.full(n, np.nan)
    volatility_padded = np.full(n, np.nan)
    change_padded[lookback-1:] = change
    volatility_padded[lookback-1:] = volatility
    
    er = np.full(n, np.nan)
    valid_er = (volatility_padded != 0) & (~np.isnan(volatility_padded))
    er[valid_er] = change_padded[valid_er] / volatility_padded[valid_er]
    
    # Scaling factor
    sc = np.full(n, np.nan)
    sc_valid = (~np.isnan(er)) & (er >= 0) & (er <= 1)
    sc[sc_valid] = (er[sc_valid] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    if n >= lookback:
        kama[lookback-1] = np.mean(close[0:lookback])
        for i in range(lookback, n):
            if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
                kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
            else:
                kama[i] = kama[i-1]
    
    # RSI calculation (14-period)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    
    if n >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[0:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[0:rsi_period])
        for i in range(rsi_period, n):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rsi = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_loss)) & (avg_loss != 0)
    rsi[valid_rsi] = 100 - (100 / (1 + avg_gain[valid_rsi] / avg_loss[valid_rsi]))
    # Handle case where avg_loss is zero (all gains)
    rsi[avg_loss == 0] = 100
    
    # Volume filter: 4h volume / 20-period average volume
    vol_ma = np.full(n, np.nan)
    if n >= 20:
        vol_ma[19] = np.mean(volume[0:20])
        for i in range(20, n):
            vol_ma[i] = (vol_ma[i-1] * 19 + volume[i]) / 20
    
    volume_ratio = np.full(n, np.nan)
    valid_vol = (~np.isnan(vol_ma)) & (vol_ma != 0)
    volume_ratio[valid_vol] = volume[valid_vol] / vol_ma[valid_vol]
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(lookback, rsi_period, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(rsi[i]) or np.isnan(volume_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: Price above KAMA (uptrend) AND RSI < 20 (oversold) AND volume confirmation
            if close[i] > kama[i] and rsi[i] < 20 and volume_ratio[i] > 1.5:
                signals[i] = 0.25
                position = 1
            # Enter short: Price below KAMA (downtrend) AND RSI > 80 (overbought) AND volume confirmation
            elif close[i] < kama[i] and rsi[i] > 80 and volume_ratio[i] > 1.5:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Price below KAMA (trend change) OR RSI > 60 (overbought exit)
            if close[i] < kama[i] or rsi[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Price above KAMA (trend change) OR RSI < 40 (oversold exit)
            if close[i] > kama[i] or rsi[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals