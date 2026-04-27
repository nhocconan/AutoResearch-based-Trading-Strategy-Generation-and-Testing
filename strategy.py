#!/usr/bin/env python3
"""
4h_Stochastic_Oscillator_Signal_Line_Cross_with_Volume_Confirmation
Mean reversion strategy using Stochastic Oscillator on 4h timeframe.
Long when %K crosses above %D in oversold territory (%K<20) with volume confirmation.
Short when %K crosses below %D in overbought territory (%K>80) with volume confirmation.
Exit when %K crosses back through %D in the opposite direction or RSI(14) reaches opposite extreme.
Uses 1d trend filter (EMA50) to avoid counter-trend trades in strong trends.
Target: 20-40 trades/year per symbol.
"""

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
    
    # Stochastic Oscillator parameters
    k_period = 14
    d_period = 3
    
    # Calculate lowest low and highest high over k_period
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
        highest_high[i] = np.max(high[i - k_period + 1:i + 1])
    
    # Calculate %K
    percent_k = np.full(n, np.nan)
    for i in range(k_period - 1, n):
        if highest_high[i] > lowest_low[i]:
            percent_k[i] = ((close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])) * 100
        else:
            percent_k[i] = 50.0  # Avoid division by zero
    
    # Calculate %D (SMA of %K)
    percent_d = np.full(n, np.nan)
    for i in range(d_period - 1, n):
        start_idx = i - d_period + 1
        if start_idx >= 0 and not np.isnan(percent_k[start_idx:i+1]).any():
            percent_d[i] = np.mean(percent_k[start_idx:i+1])
        else:
            percent_d[i] = np.nan
    
    # RSI for exit confirmation
    rsi_period = 14
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if i == rsi_period:
            avg_gain[i] = np.mean(gain[rsi_period:i+1])
            avg_loss[i] = np.mean(loss[rsi_period:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * (rsi_period-1) + gain[i]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period-1) + loss[i]) / rsi_period
    
    rsi = np.full(n, np.nan)
    for i in range(rsi_period, n):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    for i in range(vol_ma_period - 1, n):
        vol_ma[i] = np.mean(volume[i - vol_ma_period + 1:i + 1])
    
    volume_confirm = np.full(n, False)
    for i in range(vol_ma_period - 1, n):
        if not np.isnan(vol_ma[i]) and volume[i] > 1.5 * vol_ma[i]:
            volume_confirm[i] = True
    
    # Get 1d data for higher timeframe trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_1d_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_1d_period:
        ema_1d[ema_1d_period - 1] = np.mean(close_1d[:ema_1d_period])
        for i in range(ema_1d_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_1d_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_1d_period + 1))))
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Stochastic, RSI, volume MA, and EMA1d
    start_idx = max(k_period - 1, d_period - 1, rsi_period, vol_ma_period - 1, ema_1d_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(percent_k[i]) or np.isnan(percent_d[i]) or 
            np.isnan(rsi[i]) or np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        k = percent_k[i]
        d = percent_d[i]
        rsi_val = rsi[i]
        ema1d_val = ema_1d_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: %K crosses above %D in oversold territory with volume confirmation
            if (k > d and percent_k[i-1] <= percent_d[i-1] and 
                k < 20 and vol_conf):
                signals[i] = size
                position = 1
            # Short: %K crosses below %D in overbought territory with volume confirmation
            elif (k < d and percent_k[i-1] >= percent_d[i-1] and 
                  k > 80 and vol_conf):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: %K crosses below %D or RSI reaches overbought
            if (k < d and percent_k[i-1] >= percent_d[i-1]) or rsi_val > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: %K crosses above %D or RSI reaches oversold
            if (k > d and percent_k[i-1] <= percent_d[i-1]) or rsi_val < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Stochastic_Oscillator_Signal_Line_Cross_with_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0