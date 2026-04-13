#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h KAMA with 1-day RSI filter and volume confirmation.
# KAMA adapts to market noise, reducing whipsaw in ranging markets and
# tracking trends efficiently. Combined with daily RSI extremes and volume
# spikes, it captures strong moves while avoiding false signals in chop.
# Target: 15-35 trades per year (60-140 total over 4 years) for 12h timeframe.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1-day data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # RSI(14) on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d[:13] = np.nan
    
    # Align 1d RSI to 12h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10, prepend=close_12h[:10]))
    volatility = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0)
    # For simplicity, compute ER using loops (since we need prior values)
    er = np.full(len(close_12h), np.nan)
    for i in range(10, len(close_12h)):
        change_val = np.abs(close_12h[i] - close_12h[i-10])
        volatility_val = np.sum(np.abs(np.diff(close_12h[i-9:i+1])))
        if volatility_val > 0:
            er[i] = change_val / volatility_val
        else:
            er[i] = 1.0
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.full(len(close_12h), np.nan)
    kama[0] = close_12h[0]
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # Align 12h KAMA to 12h timeframe (identity, but for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Average volume (24-period = 12 hours) for volume confirmation
    avg_volume = np.full(n, np.nan)
    for i in range(24, n):
        avg_volume[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(24, n):
        # Skip if any required data is not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(avg_volume[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_aligned[i]
        rsi_val = rsi_1d_aligned[i]
        vol = volume[i]
        avg_vol = avg_volume[i]
        
        # Volume confirmation: current volume > 2.0x average volume
        volume_confirm = vol > 2.0 * avg_vol
        
        if position == 0:
            # Long: Price above KAMA + RSI < 30 (oversold) + volume confirmation
            if (price > kama_val and
                rsi_val < 30 and
                volume_confirm):
                position = 1
                signals[i] = position_size
            # Short: Price below KAMA + RSI > 70 (overbought) + volume confirmation
            elif (price < kama_val and
                  rsi_val > 70 and
                  volume_confirm):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below KAMA or RSI > 70
            if (price < kama_val or rsi_val > 70):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: Price crosses above KAMA or RSI < 30
            if (price > kama_val or rsi_val < 30):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "12h_1d_KAMA_RSI_Volume"
timeframe = "12h"
leverage = 1.0