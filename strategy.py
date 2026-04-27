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
    
    # Get daily data for ATR and volume analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 14-day ATR
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr_1d = np.concatenate([[high_1d[0] - low_1d[0]], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14d = np.full(len(tr_1d), np.nan)
    for i in range(14, len(tr_1d)):
        atr_14d[i] = np.mean(tr_1d[i-14:i])
    
    atr_14d_aligned = align_htf_to_ltf(prices, df_1d, atr_14d)
    
    # Calculate 20-day volume average
    vol_ma_20d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20d[i] = np.mean(volume_1d[i-20:i])
    
    vol_ma_20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20d)
    
    # Calculate 4-day RSI on daily closes (for momentum)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    for i in range(14, len(close_1d)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_4d = 100 - (100 / (1 + rs))
    rsi_4d_aligned = align_htf_to_ltf(prices, df_1d, rsi_4d)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup: need all indicators
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        if (np.isnan(atr_14d_aligned[i]) or 
            np.isnan(vol_ma_20d_aligned[i]) or 
            np.isnan(rsi_4d_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20d_aligned[i] if vol_ma_20d_aligned[i] > 0 else 0
        
        # Volume confirmation: > 1.5x average daily volume (moderate threshold)
        volume_confirmation = vol_ratio > 1.5
        
        # Momentum filter: RSI between 30 and 70 to avoid extremes
        momentum_ok = (rsi_4d_aligned[i] >= 30) and (rsi_4d_aligned[i] <= 70)
        
        if position == 0:
            # Long: price > close + 0.5 * ATR with volume and momentum
            if (volume_confirmation and 
                momentum_ok and 
                price > close[i-1] + 0.5 * atr_14d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price < close - 0.5 * ATR with volume and momentum
            elif (volume_confirmation and 
                  momentum_ok and 
                  price < close[i-1] - 0.5 * atr_14d_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: price < close - ATR or momentum breaks down
            if (price < close[i-1] - atr_14d_aligned[i] or 
                rsi_4d_aligned[i] < 30):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # Maintain position
        elif position == -1:
            # Short exit: price > close + ATR or momentum breaks up
            if (price > close[i-1] + atr_14d_aligned[i] or 
                rsi_4d_aligned[i] > 70):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # Maintain position
    
    return signals

name = "4h_VolumeMomentum_ATRBreakout_1dATR14_Volume20_RSI4d"
timeframe = "4h"
leverage = 1.0