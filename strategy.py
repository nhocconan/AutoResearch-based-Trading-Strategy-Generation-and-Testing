#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 14-period RSI for mean reversion
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0)
    loss = np.maximum(-delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.full(n, np.nan)
    valid_rsi = (~np.isnan(avg_gain)) & (~np.isnan(avg_loss)) & (avg_loss > 0)
    rs[valid_rsi] = avg_gain[valid_rsi] / avg_loss[valid_rsi]
    rsi_14 = np.full(n, np.nan)
    rsi_14[valid_rsi] = 100 - (100 / (1 + rs[valid_rsi]))
    
    # Get daily data for Bollinger Bands
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate Bollinger Bands: SMA(20) ± 2*std(20)
    sma_20 = np.full(len(df_1d), np.nan)
    std_20 = np.full(len(df_1d), np.nan)
    for i in range(20, len(df_1d)):
        sma_20[i] = np.mean(close_1d[i-20:i])
        std_20[i] = np.std(close_1d[i-20:i])
    
    upper_band = sma_20 + 2 * std_20
    lower_band = sma_20 - 2 * std_20
    
    upper_band_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_band_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    
    # Get 12h data for trend filter: EMA(50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50 = np.full(len(df_12h), np.nan)
    alpha = 2 / (50 + 1)
    for i in range(len(close_12h)):
        if i < 49:
            ema_50[i] = np.mean(close_12h[:i+1]) if i > 0 else close_12h[i]
        else:
            if np.isnan(ema_50[i-1]):
                ema_50[i] = np.mean(close_12h[i-49:i+1])
            else:
                ema_50[i] = close_12h[i] * alpha + ema_50[i-1] * (1 - alpha)
    
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(14, 20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_14[i]) or 
            np.isnan(upper_band_aligned[i]) or
            np.isnan(lower_band_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price touches lower Bollinger Band + RSI oversold + 12h uptrend
            if (price <= lower_band_aligned[i] and 
                rsi_14[i] < 30 and 
                ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
                signals[i] = 0.25
                position = 1
            # Short: Price touches upper Bollinger Band + RSI overbought + 12h downtrend
            elif (price >= upper_band_aligned[i] and 
                  rsi_14[i] > 70 and 
                  ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price touches upper Bollinger Band or RSI overbought
            if (price >= upper_band_aligned[i] or 
                rsi_14[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price touches lower Bollinger Band or RSI oversold
            if (price <= lower_band_aligned[i] or 
                rsi_14[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_BollingerBandTouch_RSI14_12hEMA50_v1"
timeframe = "4h"
leverage = 1.0