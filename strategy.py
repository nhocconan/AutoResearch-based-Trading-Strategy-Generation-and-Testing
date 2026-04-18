#!/usr/bin/env python3
"""
4h_1d_KAMA_Direction_RSI20_Extremes
Hypothesis: In 4h timeframe, go long when KAMA indicates uptrend and RSI(14) < 20 (oversold), short when KAMA indicates downtrend and RSI > 80 (overbought). Uses 1d ADX(14) > 20 to confirm trending market and avoid ranging conditions. Position size 0.25 targeting ~30 trades/year. Works in bull/bear by fading extremes only when trend and volatility regime align.
"""

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
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # KAMA calculation (ER=10, slow=2, fast=30)
    kama_period = 10
    fast_ema = 2
    slow_ema = 30
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, kama_period))
    abs_change = np.abs(np.diff(close))
    er_num = change[kama_period-1:]
    er_den = np.zeros_like(abs_change)
    for i in range(len(abs_change)):
        if i + kama_period <= len(abs_change):
            er_den[i] = np.sum(abs_change[i:i+kama_period])
    er = np.zeros_like(close)
    er[kama_period-1:] = np.where(er_den[kama_period-1:] != 0, er_num / er_den[kama_period-1:], 0)
    
    # Smoothing constants
    sc = (er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1)) ** 2
    
    # KAMA
    kama = np.zeros_like(close)
    kama[kama_period-1] = close[kama_period-1]
    for i in range(kama_period, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # RSI(14)
    rsi_period = 14
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close)
    avg_loss = np.zeros_like(close)
    
    if len(close) >= rsi_period:
        avg_gain[rsi_period-1] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period-1] = np.mean(loss[:rsi_period])
        for i in range(rsi_period, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i-1]) / rsi_period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # 1d ADX(14) for trend strength
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.concatenate([[np.max([high_1d[0], low_1d[0]])], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # Directional Movement
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    
    # Smoothed values
    atr_period = 14
    atr = np.zeros_like(tr)
    if len(tr) >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    dm_plus_smooth = np.zeros_like(dm_plus)
    dm_minus_smooth = np.zeros_like(dm_minus)
    if len(dm_plus) >= atr_period:
        dm_plus_smooth[atr_period-1] = np.mean(dm_plus[:atr_period])
        dm_minus_smooth[atr_period-1] = np.mean(dm_minus[:atr_period])
        for i in range(atr_period, len(dm_plus)):
            dm_plus_smooth[i] = (dm_plus_smooth[i-1] * (atr_period-1) + dm_plus[i]) / atr_period
            dm_minus_smooth[i] = (dm_minus_smooth[i-1] * (atr_period-1) + dm_minus[i]) / atr_period
    
    # DI and DX
    di_plus = np.where(atr != 0, 100 * dm_plus_smooth / atr, 0)
    di_minus = np.where(atr != 0, 100 * dm_minus_smooth / atr, 0)
    dx = np.where((di_plus + di_minus) != 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    
    # ADX
    adx = np.zeros_like(dx)
    if len(dx) >= atr_period:
        adx[atr_period-1] = np.mean(dx[:atr_period])
        for i in range(atr_period, len(dx)):
            adx[i] = (adx[i-1] * (atr_period-1) + dx[i]) / atr_period
    
    # Align 1d indicators to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, kama_period, rsi_period, atr_period)
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(adx_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend and regime filters
        kama_trend = close[i] > kama_aligned[i]  # Price above KAMA = uptrend
        rsi_oversold = rsi_aligned[i] < 20
        rsi_overbought = rsi_aligned[i] > 80
        strong_trend = adx_aligned[i] > 20  # ADX > 20 indicates trending market
        
        if position == 0:
            # Long: uptrend + oversold + trending market
            if kama_trend and rsi_oversold and strong_trend:
                signals[i] = 0.25
                position = 1
            # Short: downtrend + overbought + trending market
            elif not kama_trend and rsi_overbought and strong_trend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: trend reversal or overbought
            if not kama_trend or rsi_aligned[i] > 70:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: trend reversal or oversold
            if kama_trend or rsi_aligned[i] < 30:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1d_KAMA_Direction_RSI20_Extremes"
timeframe = "4h"
leverage = 1.0