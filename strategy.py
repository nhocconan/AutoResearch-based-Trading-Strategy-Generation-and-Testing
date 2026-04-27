#!/usr/bin/env python3
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
    
    # Get 1d data for calculations (called ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1-day Exponential Moving Average (21-period) for trend
    close_1d = df_1d['close'].values
    ema_21_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 21:
        multiplier = 2 / (21 + 1)
        ema_21_1d[20] = np.mean(close_1d[:21])
        for i in range(21, len(close_1d)):
            ema_21_1d[i] = (close_1d[i] * multiplier) + (ema_21_1d[i-1] * (1 - multiplier))
    
    # Calculate 1-day RSI (14-period) for momentum/overbought-oversold
    rsi_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[14] = np.mean(gain[:14])
        avg_loss[14] = np.mean(loss[:14])
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
        rsi_14_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1-day ATR (14-period) for volatility
    atr_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        tr1 = high[1:] - low[1:]
        tr2 = np.abs(high[1:] - close_1d[:-1])
        tr3 = np.abs(low[1:] - close_1d[:-1])
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr = np.concatenate([[np.nan], tr])
        atr_14_1d[14] = np.mean(tr[1:15])
        for i in range(15, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Align 1d indicators to 4h timeframe
    ema_21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_21_1d)
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    # Calculate 4-period volume average for spike detection
    vol_ma = np.full(n, np.nan)
    vol_period = 4
    for i in range(vol_period, n):
        vol_ma[i] = np.mean(volume[i-vol_period:i])
    
    signals = np.zeros(n)
    position = 0
    size = 0.25
    
    # Warmup period
    start_idx = max(21, vol_period) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(ema_21_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        
        # Volume spike filter: at least 1.5x average volume
        vol_filter = vol_ratio > 1.5
        
        if position == 0:
            # Long: Price above EMA21, RSI < 70 (not overbought), and breaks above EMA21 with volume
            if price > ema_21_1d_aligned[i] and rsi_14_1d_aligned[i] < 70 and vol_filter:
                signals[i] = size
                position = 1
            # Short: Price below EMA21, RSI > 30 (not oversold), and breaks below EMA21 with volume
            elif price < ema_21_1d_aligned[i] and rsi_14_1d_aligned[i] > 30 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below EMA21 or RSI > 75 (overbought)
            if price < ema_21_1d_aligned[i] or rsi_14_1d_aligned[i] > 75:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price crosses above EMA21 or RSI < 25 (oversold)
            if price > ema_21_1d_aligned[i] or rsi_14_1d_aligned[i] < 25:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA21_RSI14_Volume"
timeframe = "4h"
leverage = 1.0