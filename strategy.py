#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_And_Volume_Filter
Hypothesis: Daily Kaufman Adaptive Moving Average (KAMA) trend direction combined with RSI momentum filter and volume confirmation.
Goes long when KAMA slope is positive, RSI > 50 (bullish momentum), and volume > 1.5x 20-day average.
Goes short when KAMA slope is negative, RSI < 50 (bearish momentum), and volume > 1.5x 20-day average.
Uses weekly trend filter to avoid counter-trend trades in strong trends.
Designed for low frequency (~10-20 trades/year) with strong performance in both bull and bear markets by adapting to market conditions.
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
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly KAMA trend filter
    close_1w = df_1w['close'].values
    kama_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 30:
        # ER (Efficiency Ratio) for KAMA
        change = np.abs(np.diff(close_1w, 9))  # 9-period change
        volatility = np.sum(np.abs(np.diff(close_1w)), axis=1)  # 9-period volatility
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
        # Initialize KAMA
        kama_1w[9] = close_1w[9]  # start at index 9
        for i in range(10, len(close_1w)):
            kama_1w[i] = kama_1w[i-1] + sc[i] * (close_1w[i] - kama_1w[i-1])
    
    # Get daily data for KAMA, RSI, and volume
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily KAMA
    close_1d = df_1d['close'].values
    kama_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 30:
        # ER (Efficiency Ratio) for KAMA
        change = np.abs(np.diff(close_1d, 9))  # 9-period change
        volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 9-period volatility
        er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
        # Smoothing constants
        sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
        # Initialize KAMA
        kama_1d[9] = close_1d[9]  # start at index 9
        for i in range(10, len(close_1d)):
            kama_1d[i] = kama_1d[i-1] + sc[i] * (close_1d[i] - kama_1d[i-1])
    
    # Calculate daily RSI(14)
    rsi_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate daily volume moving average
    vol_ma_1d = np.full(len(volume), np.nan)
    for i in range(20, len(volume)):
        vol_ma_1d[i] = np.mean(volume[i-20:i])
    
    # Align weekly KAMA to daily timeframe
    kama_1w_aligned = align_htf_to_ltf(close_1d, df_1w, kama_1w)
    
    # Align daily indicators to price timeframe (1d to 1d is 1:1, but we still align for consistency)
    kama_1d_aligned = align_htf_to_ltf(close, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(close, df_1d, rsi_1d)
    vol_ma_1d_aligned = align_htf_to_ltf(close, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # KAMA(30) and volume MA(20)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1w_aligned[i]) or np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike: current volume > 1.5 x 20-day average
        vol_spike = volume[i] > (vol_ma_1d_aligned[i] * 1.5)
        
        if position == 0:
            # Long: KAMA rising (bullish trend), RSI > 50 (bullish momentum), volume spike
            if (kama_1d_aligned[i] > kama_1d_aligned[i-1] and 
                rsi_1d_aligned[i] > 50 and vol_spike):
                # Weekly trend filter: only take longs if weekly KAMA is rising
                if kama_1w_aligned[i] > kama_1w_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            # Short: KAMA falling (bearish trend), RSI < 50 (bearish momentum), volume spike
            elif (kama_1d_aligned[i] < kama_1d_aligned[i-1] and 
                  rsi_1d_aligned[i] < 50 and vol_spike):
                # Weekly trend filter: only take shorts if weekly KAMA is falling
                if kama_1w_aligned[i] < kama_1w_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down OR RSI < 40 (momentum loss)
            if (kama_1d_aligned[i] < kama_1d_aligned[i-1] or 
                rsi_1d_aligned[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up OR RSI > 60 (momentum loss)
            if (kama_1d_aligned[i] > kama_1d_aligned[i-1] or 
                rsi_1d_aligned[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_And_Volume_Filter"
timeframe = "1d"
leverage = 1.0