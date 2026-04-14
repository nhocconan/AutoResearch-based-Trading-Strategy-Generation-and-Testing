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
    
    # Load 4h data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h EMA(20) for trend direction
    close_4h = df_4h['close']
    ema_4h = pd.Series(close_4h).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    
    # 1d RSI(14) for mean reversion
    delta_1d = pd.Series(df_1d['close']).diff()
    gain_1d = delta_1d.where(delta_1d > 0, 0).rolling(window=14, min_periods=14).mean()
    loss_1d = (-delta_1d.where(delta_1d < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs_1d = gain_1d / loss_1d
    rsi_1d = (100 - (100 / (1 + rs_1d))).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # 1h volume filter: 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    position_size = 0.20
    
    for i in range(100, n):
        # Skip if any critical data is NaN
        if np.isnan(ema_4h_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]):
            continue
        
        # Session filter
        hour = hours[i]
        if hour < 8 or hour > 20:
            if position != 0:
                position = 0
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 4h uptrend (price > EMA20) + 1d RSI oversold (< 30) + volume confirmation
            if (close[i] > ema_4h_aligned[i] and 
                rsi_1d_aligned[i] < 30 and 
                volume[i] > vol_ma[i] * 1.5):
                position = 1
                signals[i] = position_size
            # Short: 4h downtrend (price < EMA20) + 1d RSI overbought (> 70) + volume confirmation
            elif (close[i] < ema_4h_aligned[i] and 
                  rsi_1d_aligned[i] > 70 and 
                  volume[i] > vol_ma[i] * 1.5):
                position = -1
                signals[i] = -position_size
        elif position == 1:
            # Exit: 4h trend reversal or RSI returns to neutral
            if close[i] < ema_4h_aligned[i] or rsi_1d_aligned[i] > 50:
                position = 0
                signals[i] = 0.0
        elif position == -1:
            # Exit: 4h trend reversal or RSI returns to neutral
            if close[i] > ema_4h_aligned[i] or rsi_1d_aligned[i] < 50:
                position = 0
                signals[i] = 0.0
    
    return signals

name = "1h_4h1d_EMA_RSI_Volume"
timeframe = "1h"
leverage = 1.0