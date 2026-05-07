#!/usr/bin/env python3
name = "4h_Combined_RSI_Momentum_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # RSI(14) on 4h close
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ma = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    loss_ma = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = gain_ma / (loss_ma + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # 12h EMA50 for trend filter
    ema_50_12h_raw = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h = align_htf_to_ltf(prices, df_12h, ema_50_12h_raw)
    
    # Volume spike (2x 20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(rsi[i]) or np.isnan(ema_50_12h[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 2.0
        
        if position == 0:
            # Long: RSI > 60 (momentum) + price above EMA50 (uptrend) + volume spike
            if rsi[i] > 60 and close[i] > ema_50_12h[i] and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: RSI < 40 (weakness) + price below EMA50 (downtrend) + volume spike
            elif rsi[i] < 40 and close[i] < ema_50_12h[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI < 50 or price below EMA50
            if rsi[i] < 50 or close[i] < ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI > 50 or price above EMA50
            if rsi[i] > 50 or close[i] > ema_50_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals