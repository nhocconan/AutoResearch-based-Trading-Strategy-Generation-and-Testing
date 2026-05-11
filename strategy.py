#!/usr/bin/env python3
name = "12h_KAMA_Trend_RSI_MeanReversion"
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
    
    # Get 1d data for KAMA trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=1)  # needs correction
    # Recalculate volatility properly
    volatility = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        volatility[i] = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
    # Avoid division by zero
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if volatility[i] > 0:
            er[i] = change[i-10] / volatility[i]
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # for EMA(2)
    slow_sc = 2 / (30 + 1)  # for EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 12h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume filter: 20-period average on 12h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold - avoid low-volume false signals
        volume_surge = vol_ratio[i] > 1.3
        
        if position == 0:
            # Long: Price > KAMA and RSI < 40 (mean reversion from oversold) with volume
            if (close[i] > kama_aligned[i] and 
                rsi_aligned[i] < 40 and 
                volume_surge):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA and RSI > 60 (mean reversion from overbought) with volume
            elif (close[i] < kama_aligned[i] and 
                  rsi_aligned[i] > 60 and 
                  volume_surge):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to KAMA or RSI reaches extreme
            if position == 1:
                # Exit long: price crosses below KAMA or RSI > 60
                if close[i] <= kama_aligned[i] or rsi_aligned[i] > 60:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above KAMA or RSI < 40
                if close[i] >= kama_aligned[i] or rsi_aligned[i] < 40:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals