#!/usr/bin/env python3
"""
4h_Bollinger_Band_Momentum_1dTrend_Volume
Hypothesis: 4h Bollinger Band breakout in direction of 1d EMA200 trend with volume confirmation.
Bollinger Bands provide dynamic support/resistance that adapts to volatility.
EMA200 trend filter ensures alignment with long-term direction, working in both bull and bear markets.
Volume confirmation reduces false breakouts. Target: 20-40 trades/year.
"""

name = "4h_Bollinger_Band_Momentum_1dTrend_Volume"
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
    
    # 1d EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_200_1d[i-1]
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Bollinger Bands (20, 2.0) on 4h
    bb_period = 20
    bb_mult = 2.0
    sma = np.full(n, np.nan)
    std = np.full(n, np.nan)
    for i in range(bb_period, n):
        sma[i] = np.mean(close[i-bb_period:i])
        std[i] = np.std(close[i-bb_period:i])
    upper_band = sma + bb_mult * std
    lower_band = sma - bb_mult * std
    
    # Volume spike: current volume > 1.5x average volume (20-period)
    vol_sma = np.full(n, np.nan)
    for i in range(20, n):
        vol_sma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(bb_period, 20, 200)  # Bollinger + volume + trend warmup
    
    for i in range(start_idx, n):
        if np.isnan(ema_200_1d_aligned[i]) or np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(vol_sma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        volume_confirm = volume[i] > 1.5 * vol_sma[i]
        
        if position == 0:
            # Long: Break above upper band and above 1d EMA200
            if close[i] > upper_band[i] and close[i] > ema_200_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band and below 1d EMA200
            elif close[i] < lower_band[i] and close[i] < ema_200_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Close below middle band (SMA)
            if close[i] < sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Close above middle band (SMA)
            if close[i] > sma[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals