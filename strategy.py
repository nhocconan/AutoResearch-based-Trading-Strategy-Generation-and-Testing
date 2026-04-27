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
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d RSI(14) for trend filter
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Calculate 1d ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_shift = np.roll(close_1d, 1)
    close_1d_shift[0] = close_1d[0]
    
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d_shift)
    tr3 = np.abs(low_1d - close_1d_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr_1d = np.full(len(tr), np.nan)
    if len(tr) >= 14:
        atr_1d[13] = np.mean(tr[1:14])
        for i in range(14, len(tr)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 1-hour ATR(14) for entry trigger
    tr_1h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_1h[0] = high[0] - low[0]
    atr_1h = np.full(n, np.nan)
    if n >= 14:
        atr_1h[13] = np.mean(tr_1h[1:14])
        for i in range(14, n):
            atr_1h[i] = (atr_1h[i-1] * 13 + tr_1h[i]) / 14
    
    # Align 1d indicators to 1h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0
    size = 0.20
    
    # Warmup period
    start_idx = max(14, 14) + 5
    
    for i in range(start_idx, n):
        if (np.isnan(rsi_1d_aligned[i]) or np.isnan(atr_1d_aligned[i]) or 
            np.isnan(atr_1h[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        price = close[i]
        volatility_ratio = atr_1h[i] / atr_1d_aligned[i] if atr_1d_aligned[i] > 0 else 0
        
        # Volatility filter: current volatility > 1.5x daily average
        vol_filter = volatility_ratio > 1.5
        
        if position == 0:
            # Long: RSI < 30 (oversold) + volatility expansion
            if rsi_1d_aligned[i] < 30 and vol_filter:
                signals[i] = size
                position = 1
            # Short: RSI > 70 (overbought) + volatility expansion
            elif rsi_1d_aligned[i] > 70 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: RSI > 50 or volatility contraction
            if rsi_1d_aligned[i] > 50 or volatility_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short exit: RSI < 50 or volatility contraction
            if rsi_1d_aligned[i] < 50 or volatility_ratio < 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1h_RSI14_Volatility_Expansion"
timeframe = "1h"
leverage = 1.0