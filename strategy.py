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
    
    # Get daily data for multi-timeframe analysis
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 50-period EMA on daily close for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Calculate 14-period ATR on daily timeframe
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    tr = np.concatenate([[np.nan], tr])
    atr_14_1d = np.full(len(tr), np.nan)
    for i in range(14, len(tr)):
        if i == 14:
            atr_14_1d[i] = np.mean(tr[1:15])
        else:
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 20-period RSI on daily timeframe
    delta = np.diff(close_1d)
    delta = np.concatenate([[np.nan], delta])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(gain, np.nan)
    avg_loss = np.full_like(loss, np.nan)
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_20_1d = 100 - (100 / (1 + rs))
    
    # Calculate 20-period volume average on daily timeframe
    vol_ma_20_1d = np.full_like(volume_1d, np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_20_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align daily indicators to 4h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    rsi_20_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_20_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    # Calculate 4-period RSI on 4h timeframe for short-term momentum
    delta_4h = np.diff(close)
    delta_4h = np.concatenate([[np.nan], delta_4h])
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.full_like(gain_4h, np.nan)
    avg_loss_4h = np.full_like(loss_4h, np.nan)
    for i in range(3, len(gain_4h)):
        if i == 3:
            avg_gain_4h[i] = np.mean(gain_4h[1:4])
            avg_loss_4h[i] = np.mean(loss_4h[1:4])
        else:
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 2 + gain_4h[i]) / 3
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 2 + loss_4h[i]) / 3
    
    rs_4h = np.divide(avg_gain_4h, avg_loss_4h, out=np.full_like(avg_gain_4h, np.nan), where=avg_loss_4h!=0)
    rsi_4_4h = 100 - (100 / (1 + rs_4h))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all indicators ready
    start_idx = max(50, 14, 20, 3) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr_14_1d_aligned[i]) or 
            np.isnan(rsi_20_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or
            np.isnan(rsi_4_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_ratio = volume[i] / vol_ma_20_1d_aligned[i] if vol_ma_20_1d_aligned[i] > 0 else 0
        atr = atr_14_1d_aligned[i]
        
        if position == 0:
            # Long: Price above daily EMA50, RSI(4) > 50 (bullish momentum), 
            # daily RSI not overbought, volume confirmation
            if (price > ema_50_1d_aligned[i] and 
                rsi_4_4h[i] > 50 and 
                rsi_20_1d_aligned[i] < 70 and 
                vol_ratio > 1.5):
                signals[i] = size
                position = 1
            # Short: Price below daily EMA50, RSI(4) < 50 (bearish momentum), 
            # daily RSI not oversold, volume confirmation
            elif (price < ema_50_1d_aligned[i] and 
                  rsi_4_4h[i] < 50 and 
                  rsi_20_1d_aligned[i] > 30 and 
                  vol_ratio > 1.5):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price closes below daily EMA50 or RSI(4) turns bearish
            if (price < ema_50_1d_aligned[i] or 
                rsi_4_4h[i] < 40):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: Price closes above daily EMA50 or RSI(4) turns bullish
            if (price > ema_50_1d_aligned[i] or 
                rsi_4_4h[i] > 60):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_EMA50_RSI4_RSI20_Volume_Filter"
timeframe = "4h"
leverage = 1.0