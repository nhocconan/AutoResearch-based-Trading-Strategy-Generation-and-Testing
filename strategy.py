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
    
    # Load daily data (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # Calculate 10-period EMA for trend (daily)
    ema_10_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 10:
        ema_10_1d[9] = np.mean(close_1d[:10])
        for i in range(10, len(close_1d)):
            ema_10_1d[i] = (close_1d[i] * 2 / (10 + 1)) + (ema_10_1d[i-1] * (10 - 1) / (10 + 1))
    
    ema_10_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_10_1d)
    
    # Calculate 14-period RSI (daily)
    rsi_14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        for i in range(14, len(close_1d)):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi_14_1d[i] = 100 - (100 / (1 + rs))
            else:
                rsi_14_1d[i] = 100
    
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # Calculate 14-period ATR (daily)
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    atr_14_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 14:
        atr_14_1d[13] = np.mean(tr[1:15])
        for i in range(15, len(close_1d)):
            atr_14_1d[i] = (atr_14_1d[i-1] * 13 + tr[i]) / 14
    
    atr_14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_14_1d)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(ema_10_1d_aligned[i]) or 
            np.isnan(rsi_14_1d_aligned[i]) or 
            np.isnan(atr_14_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above EMA10, RSI > 50, volume above average
            vol_ma = np.mean(volume[max(0, i-19):i+1]) if i >= 19 else np.mean(volume[:i+1])
            if (close[i] > ema_10_1d_aligned[i] and
                rsi_14_1d_aligned[i] > 50 and
                volume[i] > vol_ma * 1.5):
                position = 1
                signals[i] = position_size
            # Short: Price below EMA10, RSI < 50, volume above average
            elif (close[i] < ema_10_1d_aligned[i] and
                  rsi_14_1d_aligned[i] < 50 and
                  volume[i] > vol_ma * 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below EMA10 OR RSI < 40
            if (close[i] < ema_10_1d_aligned[i] or 
                rsi_14_1d_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above EMA10 OR RSI > 60
            if (close[i] > ema_10_1d_aligned[i] or 
                rsi_14_1d_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_EMA10_RSI14_Volume"
timeframe = "4h"
leverage = 1.0