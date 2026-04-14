#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 24:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate 4h RSI for momentum filter
    close_4h = df_4h['close'].values
    delta_4h = np.diff(close_4h)
    gain_4h = np.where(delta_4h > 0, delta_4h, 0)
    loss_4h = np.where(delta_4h < 0, -delta_4h, 0)
    
    avg_gain_4h = np.full_like(close_4h, np.nan)
    avg_loss_4h = np.full_like(close_4h, np.nan)
    if len(close_4h) >= 14:
        avg_gain_4h[13] = np.mean(gain_4h[1:14])
        avg_loss_4h[13] = np.mean(loss_4h[1:14])
        for i in range(14, len(close_4h)):
            avg_gain_4h[i] = (avg_gain_4h[i-1] * 13 + gain_4h[i]) / 14
            avg_loss_4h[i] = (avg_loss_4h[i-1] * 13 + loss_4h[i]) / 14
    
    rs_4h = np.where(avg_loss_4h != 0, avg_gain_4h / avg_loss_4h, 0)
    rsi_4h = 100 - (100 / (1 + rs_4h))
    rsi_4h_aligned = align_htf_to_ltf(prices, df_4h, rsi_4h)
    
    # Calculate 1d ATR for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(
            high_1d[i] - low_1d[i],
            abs(high_1d[i] - close_1d[i-1]),
            abs(low_1d[i] - close_1d[i-1])
        )
    
    atr_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 14:
        atr_1d[13] = np.mean(tr_1d[1:14])
        for i in range(14, len(df_1d)):
            atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # Calculate 1d EMA for trend filter
    ema_1d = np.full(len(df_1d), np.nan)
    if len(df_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(df_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Pre-compute hour filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.20  # 20% position size
    
    for i in range(20, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            signals[i] = 0.0
            continue
        
        # Skip if any critical data is NaN
        if (np.isnan(rsi_4h_aligned[i]) or 
            np.isnan(atr_1d_aligned[i]) or
            np.isnan(ema_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volatility filter: avoid low volatility periods
        if atr_1d_aligned[i] < 0.003 * close[i]:
            signals[i] = 0.0
            continue
        
        # Trend filter: price above/below 1d EMA
        if close[i] > ema_1d_aligned[i]:
            trend = 1  # uptrend
        else:
            trend = -1  # downtrend
        
        # Momentum filter: RSI not in extreme territory
        if rsi_4h_aligned[i] < 30 or rsi_4h_aligned[i] > 70:
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI bullish divergence + price above EMA
            if (trend == 1 and 
                rsi_4h_aligned[i] > 50 and 
                rsi_4h_aligned[i] < rsi_4h_aligned[i-1]):  # RSI falling but still bullish
                position = 1
                signals[i] = position_size
            # Short: RSI bearish divergence + price below EMA
            elif (trend == -1 and 
                  rsi_4h_aligned[i] < 50 and 
                  rsi_4h_aligned[i] > rsi_4h_aligned[i-1]):  # RSI rising but still bearish
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit: RSI overbought or trend change
            if (rsi_4h_aligned[i] > 70 or 
                close[i] < ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit: RSI oversold or trend change
            if (rsi_4h_aligned[i] < 30 or 
                close[i] > ema_1d_aligned[i]):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "1h_4h_1d_RSI_EMA_Filter"
timeframe = "1h"
leverage = 1.0