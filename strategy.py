#!/usr/bin/env python3
"""
4h_Price_Action_Reversal_1dTrend
Hypothesis: Price action reversal at 1d support/resistance with volume confirmation works in both bull and bear markets.
In bull markets, price finds support at 1d lows and bounces; in bear markets, price finds resistance at 1d highs and reverses.
Volume confirms genuine rejection of levels. Uses 1d high/low as dynamic support/resistance.
Target: 20-50 total trades over 4 years (5-12/year) to minimize fee drag.
"""

name = "4h_Price_Action_Reversal_1dTrend"
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
    
    # Get 1d data for support/resistance levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 1d EMA20 for trend filter
    ema20_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema20_1d[19] = np.mean(close_1d[:20])
        alpha = 2 / (20 + 1)
        for i in range(20, len(close_1d)):
            ema20_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema20_1d[i-1]
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # 1d volume SMA20 for volume confirmation
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # 1d high and low as support/resistance levels
    res_1d = high_1d  # 1d high as resistance
    sup_1d = low_1d   # 1d low as support
    res_1d_aligned = align_htf_to_ltf(prices, df_1d, res_1d)
    sup_1d_aligned = align_htf_to_ltf(prices, df_1d, sup_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 30)  # warmup for EMA and enough price history
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(res_1d_aligned[i]) or np.isnan(sup_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x average 4h volume from 1d
        vol_4h_approx = vol_sma20_1d_aligned[i] / 6.0  # 24h/4h = 6
        volume_confirm = volume[i] > 1.5 * vol_4h_approx
        
        if position == 0:
            # Long: Price rejects 1d low (support) with volume in uptrend
            if low[i] <= sup_1d_aligned[i] * 1.001 and close[i] > sup_1d_aligned[i] and \
               close[i] > ema20_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Price rejects 1d high (resistance) with volume in downtrend
            elif high[i] >= res_1d_aligned[i] * 0.999 and close[i] < res_1d_aligned[i] and \
                 close[i] < ema20_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price breaks below 1d low or trend turns down
            if low[i] < sup_1d_aligned[i] * 0.999 or close[i] < ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price breaks above 1d high or trend turns up
            if high[i] > res_1d_aligned[i] * 1.001 or close[i] > ema20_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals