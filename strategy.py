#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and RSI (longer-term context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 20-period ATR using Wilder's smoothing (for volatility filter)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_1d = np.full(len(tr), np.nan)
    for i in range(20, len(tr)):
        if i == 20:
            atr_1d[i] = np.mean(tr[1:21])
        else:
            atr_1d[i] = (atr_1d[i-1] * 19 + tr[i]) / 20
    
    # Calculate 14-period RSI for mean reversion signals
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Get 4h data for EMA200 trend filter (long-term trend)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 200:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA200
    ema_period = 200
    ema_4h = np.full(len(close_4h), np.nan)
    if len(close_4h) >= ema_period:
        ema_4h[ema_period - 1] = np.mean(close_4h[:ema_period])
        for i in range(ema_period, len(close_4h)):
            ema_4h[i] = (close_4h[i] * (2 / (ema_period + 1)) + 
                        ema_4h[i-1] * (1 - (2 / (ema_period + 1))))
    
    # Align indicators to 4h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # Volume filter: current volume > 1.8x 20-period average
    vol_ma = np.full(n, np.nan)
    vol_period = 20
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, RSI, EMA, and volume MA
    start_idx = max(20, 200, vol_period) + 30  # extra buffer for stability
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_4h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        atr = atr_1d_aligned[i]
        
        if position == 0:
            # Long: RSI < 25 (deeply oversold) + volume spike + price > 4h EMA200
            if (rsi_1d_aligned[i] < 25 and 
                vol_ratio > 1.8 and 
                price > ema_4h_aligned[i]):
                signals[i] = size
                position = 1
            # Short: RSI > 75 (deeply overbought) + volume spike + price < 4h EMA200
            elif (rsi_1d_aligned[i] > 75 and 
                  vol_ratio > 1.8 and 
                  price < ema_4h_aligned[i]):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 55 (mean reversion) OR ATR-based stop
            if (rsi_1d_aligned[i] > 55 or 
                price < ema_4h_aligned[i] - 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 45 (mean reversion) OR ATR-based stop
            if (rsi_1d_aligned[i] < 45 or 
                price > ema_4h_aligned[i] + 2.0 * atr):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_RSI_Extreme_Reversion_Volume_EMA200"
timeframe = "4h"
leverage = 1.0