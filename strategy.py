#!/usr/bin/env python3
"""
12h_1d_1w_KAMA_Reversal_v1
Hypothesis: On 12h timeframe, use Kaufman Adaptive Moving Average (KAMA) with ER=10 to identify trend direction.
Enter long when price crosses above KAMA with daily price above weekly VWAP (bull regime), short when price crosses below KAMA with daily price below weekly VWAP (bear regime).
Exit when price crosses back below/above KAMA. Uses volume confirmation (1.5x 24-period average) to avoid false signals.
Designed for low trade frequency (15-30/year) by requiring trend alignment across 12h, 1d, and 1w timeframes.
Works in bull/bear via weekly VWAP regime filter and adaptive KAMA that reduces whipsaws in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_1w_KAMA_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY VWAP FOR REGIME FILTER ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Calculate typical price and VWAP
    typical_price_1w = (high_1w + low_1w + close_1w) / 3
    vp_1w = typical_price_1w * volume_1w
    cum_vp_1w = np.nancumsum(vp_1w)
    cum_vol_1w = np.nancumsum(volume_1w)
    vwap_1w = np.divide(cum_vp_1w, cum_vol_1w, out=np.full_like(cum_vp_1w, np.nan), where=cum_vol_1w!=0)
    
    # === DAILY PRICE FOR ADDITIONAL CONFIRMATION ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # === 12H KAMA (ER=10) ===
    # Calculate Efficiency Ratio and smoothing constants
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0])), axis=0)  # Will compute properly below
    
    # Proper ER calculation: change over 10 periods / sum of absolute changes over 10 periods
    change_10 = np.zeros(n)
    vol_10 = np.zeros(n)
    
    for i in range(n):
        if i >= 10:
            change_10[i] = np.abs(close[i] - close[i-10])
            vol_10[i] = np.sum(np.abs(np.diff(close[i-9:i+1])))
        else:
            change_10[i] = np.abs(close[i] - close[0]) if i > 0 else 0
            vol_10[i] = np.sum(np.abs(np.diff(close[0:i+1]))) if i > 0 else 0
    
    # Avoid division by zero
    er = np.divide(change_10, vol_10, out=np.zeros_like(change_10), where=vol_10!=0)
    # Smoothing constants: fastest SC = 2/(2+1) = 0.67, slowest SC = 2/(30+1) = 0.0645
    sc = (er * (0.67 - 0.0645) + 0.0645) ** 2
    
    # Calculate KAMA
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Align data to 12h timeframe
    vwap_1w_aligned = align_htf_to_ltf(prices, df_1w, vwap_1w)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)  # KAMA calculated on daily, aligned to 12h
    
    # Volume average (24-period for 12h = ~12 days) for confirmation
    vol_avg = np.zeros(n)
    vol_sum = 0.0
    vol_count = 0
    for i in range(n):
        vol_sum += volume[i]
        vol_count += 1
        if i >= 24:
            vol_sum -= volume[i-24]
            vol_count -= 1
        if vol_count > 0:
            vol_avg[i] = vol_sum / vol_count
        else:
            vol_avg[i] = 0.0
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # start after warmup
        # Skip if indicators not available
        if (np.isnan(vwap_1w_aligned[i]) or np.isnan(close_1d_aligned[i]) or 
            np.isnan(kama_aligned[i]) or vol_avg[i] == 0.0):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: at least 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_avg[i]
        
        # Regime filter: daily price above/below weekly VWAP
        price_above_vwap = close_1d_aligned[i] > vwap_1w_aligned[i]
        price_below_vwap = close_1d_aligned[i] < vwap_1w_aligned[i]
        
        # Entry conditions: price crosses KAMA with regime and volume confirmation
        long_setup = (close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]) and \
                     price_above_vwap and vol_confirm
        short_setup = (close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]) and \
                     price_below_vwap and vol_confirm
        
        # Exit conditions: price crosses back below/above KAMA
        exit_long = close[i] < kama_aligned[i] and close[i-1] >= kama_aligned[i-1]
        exit_short = close[i] > kama_aligned[i] and close[i-1] <= kama_aligned[i-1]
        
        if long_setup and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_setup and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals