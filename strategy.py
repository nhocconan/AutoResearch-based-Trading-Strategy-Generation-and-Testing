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
    
    # Load 1d data for 14-day RSI and 200-day SMA (trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 14-day RSI (Wilder's smoothing)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full_like(close_1d, np.nan)
    avg_loss = np.full_like(close_1d, np.nan)
    if len(gain) >= 14:
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi_14 = 100 - (100 / (1 + rs))
    
    # 200-day SMA
    sma_200 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        for i in range(199, len(close_1d)):
            sma_200[i] = np.mean(close_1d[i-199:i+1])
    
    # Align RSI and SMA to 4h timeframe
    rsi_14_4h = align_htf_to_ltf(prices, df_1d, rsi_14)
    sma_200_4h = align_htf_to_ltf(prices, df_1d, sma_200)
    
    # Load 4h data for entry/exit logic
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 14:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # 14-period ATR for stoploss and position sizing
    tr1 = high_4h[1:] - low_4h[1:]
    tr2 = np.abs(high_4h[1:] - close_4h[:-1])
    tr3 = np.abs(low_4h[1:] - close_4h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    atr_14 = np.full_like(close_4h, np.nan)
    if len(tr) >= 14:
        atr_14[13] = np.mean(tr[1:15])
        for i in range(14, len(tr)):
            atr_14[i] = (atr_14[i-1] * 13 + tr[i]) / 14
    
    # Align ATR to 4h (already aligned, but ensure no look-ahead)
    atr_14_4h = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_14_4h[i]) or 
            np.isnan(sma_200_4h[i]) or 
            np.isnan(atr_14_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price > 200-day SMA (bullish bias)
            if rsi_14_4h[i] < 30 and close[i] > sma_200_4h[i]:
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) and price < 200-day SMA (bearish bias)
            elif rsi_14_4h[i] > 70 and close[i] < sma_200_4h[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI > 50 (mean reversion) OR stoploss hit (2 * ATR)
            if rsi_14_4h[i] > 50 or close[i] < (signals[i-1] * sma_200_4h[i] / position_size if i>0 and signals[i-1]!=0 else close[i]) - 2 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI < 50 (mean reversion) OR stoploss hit (2 * ATR)
            if rsi_14_4h[i] < 50 or close[i] > (signals[i-1] * sma_200_4h[i] / position_size if i>0 and signals[i-1]!=0 else close[i]) + 2 * atr_14_4h[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_RSI14_SMA200_MeanReversion"
timeframe = "4h"
leverage = 1.0