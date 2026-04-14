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
    
    # Load weekly data (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly pivot points (classic)
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2 * pivot_1w - low_1w
    s1_1w = 2 * pivot_1w - high_1w
    
    pivot_1w_aligned = align_htf_to_ltf(prices, df_1w, pivot_1w)
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Calculate 10-period EMA for trend filter (weekly)
    ema10_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 10:
        ema10_1w[9] = np.mean(close_1w[:10])
        for i in range(10, len(close_1w)):
            ema10_1w[i] = close_1w[i] * 0.1818 + ema10_1w[i-1] * 0.8182  # alpha = 2/(10+1)
    
    ema10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema10_1w)
    
    # Calculate 14-period RSI for momentum (weekly)
    rsi14_1w = np.full_like(close_1w, np.nan)
    if len(close_1w) >= 14:
        delta = np.diff(close_1w, prepend=close_1w[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1w, np.nan)
        avg_loss = np.full_like(close_1w, np.nan)
        
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        
        for i in range(14, len(close_1w)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        
        rs = np.full_like(close_1w, np.nan)
        for i in range(13, len(close_1w)):
            if avg_loss[i] > 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi14_1w[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi14_1w[i] = 100 if avg_gain[i] > 0 else 0
    
    rsi14_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi14_1w)
    
    # Calculate 14-period ATR for volatility filter (weekly)
    atr14_1w = np.full_like(close_1w, np.nan)
    if len(high_1w) >= 14 and len(low_1w) >= 14 and len(close_1w) >= 14:
        tr1 = high_1w - low_1w
        tr2 = np.abs(high_1w - np.roll(close_1w, 1))
        tr3 = np.abs(low_1w - np.roll(close_1w, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]
        
        if len(close_1w) >= 14:
            atr14_1w[13] = np.mean(tr[1:14])
            for i in range(14, len(close_1w)):
                atr14_1w[i] = (atr14_1w[i-1] * 13 + tr[i]) / 14
    
    atr14_1w_aligned = align_htf_to_ltf(prices, df_1w, atr14_1w)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_1w_aligned[i]) or 
            np.isnan(r1_1w_aligned[i]) or 
            np.isnan(s1_1w_aligned[i]) or 
            np.isnan(ema10_1w_aligned[i]) or 
            np.isnan(rsi14_1w_aligned[i]) or 
            np.isnan(atr14_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume ratio: current 6h volume vs 20-period average
        vol_ma_20 = np.full_like(volume, np.nan)
        for j in range(19, len(volume)):
            vol_ma_20[j] = np.mean(volume[j-19:j+1])
        
        if np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0:
            volume_ratio = 0
        else:
            volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: Price above S1 + price above EMA10 + RSI > 55 + volume surge + ATR filter
            if (close[i] > s1_1w_aligned[i] and
                close[i] > ema10_1w_aligned[i] and
                rsi14_1w_aligned[i] > 55 and
                volume_ratio > 2.0 and
                atr14_1w_aligned[i] > 0):
                position = 1
                signals[i] = position_size
            # Short: Price below R1 + price below EMA10 + RSI < 45 + volume surge + ATR filter
            elif (close[i] < r1_1w_aligned[i] and
                  close[i] < ema10_1w_aligned[i] and
                  rsi14_1w_aligned[i] < 45 and
                  volume_ratio > 2.0 and
                  atr14_1w_aligned[i] > 0):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: Price below S1 OR RSI < 40
            if (close[i] < s1_1w_aligned[i] or 
                rsi14_1w_aligned[i] < 40):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: Price above R1 OR RSI > 60
            if (close[i] > r1_1w_aligned[i] or 
                rsi14_1w_aligned[i] > 60):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "6h_1w_Pivot_R1S1_EMA10_RSI14_Volume_ATR"
timeframe = "6h"
leverage = 1.0