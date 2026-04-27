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
    
    # Get daily data for ATR and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 14-day ATR
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr_1d = np.full(len(tr), np.nan)
    for i in range(13, len(tr)):
        if i == 13:
            atr_1d[i] = np.mean(tr[:14])
        else:
            atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Calculate 14-day RSI
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(len(gain), np.nan)
    avg_loss = np.full(len(loss), np.nan)
    
    for i in range(14, len(gain)):
        if i == 14:
            avg_gain[i] = np.mean(gain[:14])
            avg_loss[i] = np.mean(loss[:14])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi_1d = 100 - (100 / (1 + rs))
    
    # Align ATR and RSI to daily timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 50-week EMA for trend filter
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (50 + 1)) + 
                         ema_1w[i-1] * (1 - (2 / (50 + 1))))
    
    # Align weekly EMA to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need ATR, RSI, and weekly EMA
    start_idx = max(14, 49)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(atr_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        atr = atr_1d_aligned[i]
        rsi = rsi_1d_aligned[i]
        ema_trend = ema_1w_aligned[i]
        
        if position == 0:
            # Long: Oversold RSI with price above weekly EMA in uptrend
            if (rsi < 30 and price > ema_trend):
                signals[i] = size
                position = 1
            # Short: Overbought RSI with price below weekly EMA in downtrend
            elif (rsi > 70 and price < ema_trend):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RSI returns to neutral or trend fails
            if rsi > 50 or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RSI returns to neutral or trend fails
            if rsi < 50 or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_RSI_MeanReversion_WeeklyEMA_Trend"
timeframe = "1d"
leverage = 1.0