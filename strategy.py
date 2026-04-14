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
    
    # Load weekly data (HTF) once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    volume_1w = df_1w['volume'].values
    
    # Weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    
    for i in range(len(gain)):
        if i < 14:
            if i == 0:
                avg_gain[i] = gain[i]
                avg_loss[i] = loss[i]
            else:
                avg_gain[i] = (avg_gain[i-1] * i + gain[i]) / (i + 1)
                avg_loss[i] = (avg_loss[i-1] * i + loss[i]) / (i + 1)
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi_1w = 100 - (100 / (1 + rs))
    rsi_1w = np.where(np.isnan(rsi_1w), 50, rsi_1w)
    
    # Weekly ATR (14-period)
    tr = np.zeros(len(df_1w))
    tr[0] = high_1w[0] - low_1w[0]
    for i in range(1, len(df_1w)):
        tr[i] = max(
            high_1w[i] - low_1w[i],
            abs(high_1w[i] - close_1w[i-1]),
            abs(low_1w[i] - close_1w[i-1])
        )
    
    atr_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 14:
        atr_1w[13] = np.mean(tr[:14])
        for i in range(14, len(df_1w)):
            atr_1w[i] = (atr_1w[i-1] * 13 + tr[i]) / 14
    
    # Weekly volume average (20-period)
    vol_ma_1w = np.full(len(df_1w), np.nan)
    if len(df_1w) >= 20:
        for i in range(19, len(df_1w)):
            vol_ma_1w[i] = np.mean(volume_1w[i-19:i+1])
    
    # Align to daily timeframe
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    atr_1w_aligned = align_htf_to_ltf(prices, df_1w, atr_1w)
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
    # Daily ATR (14-period) for stop loss
    tr_d = np.zeros(n)
    tr_d[0] = high[0] - low[0]
    for i in range(1, n):
        tr_d[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_d = np.full(n, np.nan)
    if n >= 14:
        atr_d[13] = np.mean(tr_d[:14])
        for i in range(14, n):
            atr_d[i] = (atr_d[i-1] * 13 + tr_d[i]) / 14
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position size
    
    for i in range(30, n):
        # Skip if any critical data is NaN
        if (np.isnan(rsi_1w_aligned[i]) or
            np.isnan(atr_1w_aligned[i]) or
            np.isnan(vol_ma_1w_aligned[i]) or
            np.isnan(atr_d[i])):
            signals[i] = 0.0
            continue
        
        # Skip low volatility weeks (ATR < 2% of price)
        if atr_1w_aligned[i] / close[i] < 0.02:
            signals[i] = 0.0
            continue
        
        # Skip low volume weeks (volume < 70% of 20-week average)
        if volume[i] < vol_ma_1w_aligned[i] * 0.7:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 30 (oversold) and price above weekly low
            if rsi_1w_aligned[i] < 30 and close[i] > low_1w[i]:
                position = 1
                signals[i] = position_size
            # Short: RSI > 70 (overbought) and price below weekly high
            elif rsi_1w_aligned[i] > 70 and close[i] < high_1w[i]:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI > 50 or ATR-based stop loss
            if (rsi_1w_aligned[i] > 50 or 
                close[i] < high[i] - 2.0 * atr_d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI < 50 or ATR-based stop loss
            if (rsi_1w_aligned[i] < 50 or 
                close[i] > low[i] + 2.0 * atr_d[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1d_1w_RSI_MeanReversion_VolFilter_ATRStop"
timeframe = "1d"
leverage = 1.0