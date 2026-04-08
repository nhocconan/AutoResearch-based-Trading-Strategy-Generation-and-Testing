#!/usr/bin/env python3
# 1d_kama_roc_volume_v1
# Hypothesis: Daily trend following using KAMA trend direction, ROC momentum, and volume confirmation.
# Long when KAMA trend is up, ROC > 0, and volume > 1.5x 20-day average.
# Short when KAMA trend is down, ROC < 0, and volume > 1.5x 20-day average.
# Exit when KAMA trend reverses or volume drops below average.
# Uses KAMA for adaptive trend detection to reduce whipsaw in choppy markets.
# Target: 15-25 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_kama_roc_volume_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA parameters
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_period))  # |close - close[er_period]|
    change = np.concatenate([np.full(er_period, np.nan), change])
    
    volatility = np.abs(np.diff(close))  # |close - close[1]|
    volatility = np.concatenate([np.array([np.nan]), volatility])
    
    # Sum volatility over er_period periods
    vol_sum = np.full(n, np.nan)
    for i in range(er_period, n):
        vol_sum[i] = np.nansum(volatility[i-er_period+1:i+1])
    
    # Calculate ER
    er = np.full(n, np.nan)
    for i in range(er_period, n):
        if vol_sum[i] > 0:
            er[i] = change[i] / vol_sum[i]
        else:
            er[i] = 0
    
    # Smoothing constants
    sc = np.full(n, np.nan)
    for i in range(er_period, n):
        fast_sc = 2 / (fast_ema + 1)
        slow_sc = 2 / (slow_ema + 1)
        sc[i] = (er[i] * fast_sc + (1 - er[i]) * slow_sc) ** 2
    
    # Calculate KAMA
    kama = np.full(n, np.nan)
    kama[er_period] = close[er_period]  # Seed value
    for i in range(er_period + 1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA trend direction: 1 if KAMA rising, -1 if falling
    kama_trend = np.full(n, 0)
    for i in range(er_period + 2, n):
        if not np.isnan(kama[i]) and not np.isnan(kama[i-1]):
            if kama[i] > kama[i-1]:
                kama_trend[i] = 1
            elif kama[i] < kama[i-1]:
                kama_trend[i] = -1
    
    # ROC (Rate of Change) - 10 period
    roc_period = 10
    roc = np.full(n, np.nan)
    for i in range(roc_period, n):
        if not np.isnan(close[i]) and not np.isnan(close[i-roc_period]) and close[i-roc_period] != 0:
            roc[i] = (close[i] - close[i-roc_period]) / close[i-roc_period]
    
    # Volume filter: 1.5x 20-day average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period-1, n):
        vol_ma[i] = np.mean(volume[i-vol_ma_period+1:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(er_period + 2, roc_period, vol_ma_period) + 1
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_trend[i]) or np.isnan(roc[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: KAMA trend turns down or volume drops below average
            if kama_trend[i] == -1 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: KAMA trend turns up or volume drops below average
            if kama_trend[i] == 1 or volume[i] < vol_ma[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: KAMA trend up, ROC positive, volume surge
            if (kama_trend[i] == 1 and 
                roc[i] > 0 and 
                vol_surge[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: KAMA trend down, ROC negative, volume surge
            elif (kama_trend[i] == -1 and 
                  roc[i] < 0 and 
                  vol_surge[i]):
                position = -1
                signals[i] = -0.25
    
    return signals