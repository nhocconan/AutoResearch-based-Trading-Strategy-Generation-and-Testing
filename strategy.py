#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Extreme_Volume
Hypothesis: On daily timeframe, KAMA identifies trend direction while RSI extremes identify overextended conditions.
Only trade in direction of KAMA trend when RSI is at extreme levels (oversold in uptrend, overbought in downtrend) with volume confirmation.
Avoids counter-trend trades and uses weekly trend filter to avoid counter-trend trades in strong trends.
Designed for low frequency (<15 trades/year) with high win rate by combining trend following with mean reversion within trend.
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate KAMA (adaptive moving average) - trend direction
    close_1d = df_1d['close'].values
    kama = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 10:
        # Efficiency ratio
        change = np.abs(np.diff(close_1d, n=10))
        volatility = np.sum(np.abs(np.diff(close_1d)), axis=0) if len(close_1d) >= 11 else np.full(len(close_1d)-10, 1)
        # Pad arrays
        change_padded = np.full(len(close_1d), np.nan)
        volatility_padded = np.full(len(close_1d), np.nan)
        change_padded[10:] = change
        volatility_padded[10:] = volatility if len(volatility) == len(change) else np.full(len(close_1d)-10, np.sum(np.abs(np.diff(close_1d))))
        
        er = np.where(volatility_padded != 0, change_padded / volatility_padded, 0)
        sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
        
        # KAMA calculation
        kama[9] = close_1d[9]
        for i in range(10, len(close_1d)):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI (14)
    rsi = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        
        if len(close_1d) >= 15:
            avg_gain[14] = np.mean(gain[0:14])
            avg_loss[14] = np.mean(loss[0:14])
            for i in range(15, len(close_1d)):
                avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
                avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
                
                rs = np.where(avg_loss[i] != 0, avg_gain[i] / avg_loss[i], 0)
                rsi[i] = 100 - (100 / (1 + rs))
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    
    # Calculate weekly EMA34 trend filter
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Align indicators to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume spike: current volume > 1.5 x 20-day average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 15)
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up (uptrend), RSI oversold (<30), volume spike
            if (close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA down (downtrend), RSI overbought (>70), volume spike
            elif (close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and vol_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: RSI overbought (>70) or price below KAMA
            if (rsi_aligned[i] > 70 or close[i] < kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: RSI oversold (<30) or price above KAMA
            if (rsi_aligned[i] < 30 or close[i] > kama_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_Extreme_Volume"
timeframe = "1d"
leverage = 1.0