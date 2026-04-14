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
    
    # Load daily data (HTF) once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 60-day rolling 10th percentile of ATR(14) for volatility regime filter
    tr = np.zeros(len(df_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr[:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 60-day rolling 10th percentile of ATR (volatility regime)
    atr_percentile = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 60:
        for i in range(59, len(df_1d)):
            window = atr_1d[i-59:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                atr_percentile[i] = np.percentile(valid, 10)
    
    atr_percentile_6h = align_htf_to_ltf(prices, df_1d, atr_percentile)
    atr_6h = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 6h RSI(14) for mean reversion entries
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    if n >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full(n, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 60-day rolling 10th percentile of 1d RSI for regime filter
    rsi_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        delta_1d = np.diff(close_1d, prepend=close_1d[0])
        gain_1d = np.where(delta_1d > 0, delta_1d, 0)
        loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
        avg_gain_1d = np.full(len(df_1d), np.nan)
        avg_loss_1d = np.full(len(df_1d), np.nan)
        avg_gain_1d[13] = np.mean(gain_1d[:14])
        avg_loss_1d[13] = np.mean(loss_1d[:14])
        for i in range(14, len(df_1d)):
            avg_gain_1d[i] = (avg_gain_1d[i-1] * 13 + gain_1d[i]) / 14
            avg_loss_1d[i] = (avg_loss_1d[i-1] * 13 + loss_1d[i]) / 14
        rs_1d = np.divide(avg_gain_1d, avg_loss_1d, out=np.full(len(df_1d), np.nan), where=avg_loss_1d!=0)
        rsi_1d = 100 - (100 / (1 + rs_1d))
    
    rsi_1d_percentile = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 60:
        for i in range(59, len(df_1d)):
            window = rsi_1d[i-59:i+1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 10:
                rsi_1d_percentile[i] = np.percentile(valid, 10)
    
    rsi_1d_percentile_6h = align_htf_to_ltf(prices, df_1d, rsi_1d_percentile)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(60, n):
        # Skip if any critical data is NaN
        if (np.isnan(atr_percentile_6h[i]) or
            np.isnan(atr_6h[i]) or
            np.isnan(rsi[i]) or
            np.isnan(rsi_1d_percentile_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volatility regime filter: only trade when current ATR > 10th percentile of 60-day ATR
        if atr_6h[i] <= atr_percentile_6h[i]:
            signals[i] = 0.0
            continue
        
        # Mean reversion regime filter: only trade when 1d RSI < 10th percentile of 60-day RSI (oversold conditions)
        if rsi_1d_percentile_6h[i] > 30:  # Avoid extremely oversold conditions that may persist
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) in high volatility regime
            if rsi[i] < 30:
                position = 1
                signals[i] = position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI > 70 (overbought) or volatility drops below regime threshold
            if rsi[i] > 70 or atr_6h[i] <= atr_percentile_6h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
    
    return signals

name = "6h_1d_VolRegime_MeanReversion_RSI"
timeframe = "6h"
leverage = 1.0