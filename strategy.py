#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Use KAMA direction on 1d to determine long-term trend, enter on RSI pullbacks in trend direction with volume confirmation. Weekly timeframe acts as regime filter - only trade when price is above/below weekly KAMA. Designed for low trade frequency to work in both bull and bear markets by capturing trending moves while avoiding whipsaws in ranging markets.
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
    
    # KAMA on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # 10-period volatility
    er = np.zeros_like(close_1d)
    er[10:] = change[9:] / (volatility[9:] + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Align KAMA to daily timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # RSI on daily timeframe
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[14] = np.mean(gain[1:15])
    avg_loss[14] = np.mean(loss[1:15])
    for i in range(15, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: >1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # Weekly KAMA
    change_w = np.abs(np.diff(close_1w, k=5))
    volatility_w = np.sum(np.abs(np.diff(close_1w)), axis=1)
    er_w = np.zeros_like(close_1w)
    er_w[5:] = change_w[4:] / (volatility_w[4:] + 1e-10)
    sc_w = (er_w * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama_w = np.full_like(close_1w, np.nan)
    kama_w[4] = close_1w[4]
    for i in range(5, len(close_1w)):
        kama_w[i] = kama_w[i-1] + sc_w[i] * (close_1w[i] - kama_w[i-1])
    kama_w_aligned = align_htf_to_ltf(prices, df_1w, kama_w)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Need indicators to warm up
    
    for i in range(start_idx, n):
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(kama_w_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        wkama_val = kama_w_aligned[i]
        vol_spike = volume_spike[i]
        
        if position == 0:
            # Long: price above daily KAMA, RSI < 40 (pullback), above weekly KAMA, volume spike
            if price > kama_val and rsi_val < 40 and price > wkama_val and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below daily KAMA, RSI > 60 (pullback), below weekly KAMA, volume spike
            elif price < kama_val and rsi_val > 60 and price < wkama_val and vol_spike:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            signals[i] = 0.25
            # Exit: price below daily KAMA or RSI > 70
            if price < kama_val or rsi_val > 70:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            signals[i] = -0.25
            # Exit: price above daily KAMA or RSI < 30
            if price > kama_val or rsi_val < 30:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0