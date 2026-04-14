#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using weekly EMA200 trend filter and 1d RSI mean reversion.
# Long when price > weekly EMA200 AND 1d RSI < 40 (oversold in uptrend).
# Short when price < weekly EMA200 AND 1d RSI > 60 (overbought in downtrend).
# Exit when price crosses weekly EMA200 or RSI returns to neutral (40-60).
# Weekly EMA200 provides major trend filter to avoid counter-trend trades,
# while 1d RSI captures mean reversion within the trend. Works in both bull/bear
# by only trading with the major trend. Target: 20-30 trades/year per symbol.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Load weekly data ONCE for EMA200
    df_w = get_htf_data(prices, '1w')
    if len(df_w) < 200:
        return np.zeros(n)
    
    close_w = df_w['close'].values
    
    # Calculate weekly EMA200
    ema_200_w = np.full(len(close_w), np.nan)
    if len(close_w) >= 200:
        multiplier = 2 / (200 + 1)
        ema_200_w[199] = np.mean(close_w[:200])
        for i in range(200, len(close_w)):
            ema_200_w[i] = (close_w[i] * multiplier) + (ema_200_w[i-1] * (1 - multiplier))
    
    # Load daily data ONCE for RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    rsi = np.full(len(close_1d), 50.0)  # default neutral
    
    if len(close_1d) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        
        for i in range(14, len(close_1d)):
            avg_gain[i] = (gain[i-1] + avg_gain[i-1] * 13) / 14
            avg_loss[i] = (loss[i-1] + avg_loss[i-1] * 13) / 14
            
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Load 6h data ONCE for price alignment
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 10:
        return np.zeros(n)
    
    close_6h = df_6h['close'].values
    
    # Align indicators to 6h timeframe
    ema_200_w_aligned = align_htf_to_ltf(prices, df_w, ema_200_w)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    close_6h_aligned = align_htf_to_ltf(prices, df_6h, close_6h)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    # Start after enough data for calculations
    start = max(200, 14)  # Need weekly EMA200 and daily RSI
    
    for i in range(start, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_200_w_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or
            np.isnan(close_6h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close_6h_aligned[i]
        ema200 = ema_200_w_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Look for entries in direction of weekly trend
            # Long: price above weekly EMA200 AND RSI oversold (<40)
            if price > ema200 and rsi_val < 40:
                position = 1
                signals[i] = position_size
            # Short: price below weekly EMA200 AND RSI overbought (>60)
            elif price < ema200 and rsi_val > 60:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below weekly EMA200 OR RSI returns to neutral (>50)
            if price < ema200 or rsi_val > 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price crosses above weekly EMA200 OR RSI returns to neutral (<50)
            if price > ema200 or rsi_val < 50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_EMA200_1dRSI_MeanReversion_v1"
timeframe = "6h"
leverage = 1.0