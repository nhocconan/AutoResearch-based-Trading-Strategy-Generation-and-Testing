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
    
    # Get weekly data for calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly ATR for volatility (period 14)
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1w = np.full(len(df_1w), np.nan)
    for i in range(14, len(df_1w)):
        atr_1w[i] = np.nanmean(tr[i-14:i+1])
    
    # Calculate 5-period low for Donchian channel (weekly)
    low_5 = np.full(len(df_1w), np.nan)
    for i in range(4, len(df_1w)):
        low_5[i] = np.min(low_1w[i-4:i+1])
    
    # Calculate 5-period high for Donchian channel (weekly)
    high_5 = np.full(len(df_1w), np.nan)
    for i in range(4, len(df_1w)):
        high_5[i] = np.max(high_1w[i-4:i+1])
    
    # Align weekly indicators to 6h timeframe
    low_5_aligned = align_htf_to_ltf(prices, df_1w, low_5)
    high_5_aligned = align_htf_to_ltf(prices, df_1w, high_5)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    
    # Calculate 6-period RSI for momentum filter (6h)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(6, n):
        if i == 6:
            avg_gain[i] = np.mean(gain[0:7])
            avg_loss[i] = np.mean(loss[0:7])
        else:
            avg_gain[i] = (avg_gain[i-1] * 5 + gain[i]) / 6
            avg_loss[i] = (avg_loss[i-1] * 5 + loss[i]) / 6
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(low_5_aligned[i]) or np.isnan(high_5_aligned[i]) or 
            np.isnan(atr_1w_aligned[i]) or np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: ATR > 0.5 * ATR MA (avoid low volatility choppy periods)
        atr_ma = np.full(n, np.nan)
        for j in range(29, n):
            atr_ma[j] = np.nanmean(atr_1w_aligned[j-29:j+1])
        vol_filter = atr_1w_aligned[i] > 0.5 * atr_ma[i] if not np.isnan(atr_ma[i]) else False
        
        # Momentum filter: RSI between 30 and 70 (avoid extremes)
        mom_filter = (rsi[i] >= 30) and (rsi[i] <= 70)
        
        # Entry conditions: Donchian breakout with filters
        long_breakout = (close[i] > high_5_aligned[i]) and vol_filter and mom_filter
        short_breakout = (close[i] < low_5_aligned[i]) and vol_filter and mom_filter
        
        # Exit conditions: opposite Donchian break
        long_exit = close[i] < low_5_aligned[i]
        short_exit = close[i] > high_5_aligned[i]
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_donchian_vol_filter_v1"
timeframe = "6h"
leverage = 1.0