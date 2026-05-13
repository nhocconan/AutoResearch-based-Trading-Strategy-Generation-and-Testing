#!/usr/bin/env python3
name = "1h_RSI_Momentum_Volume"
timeframe = "1h"
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
    
    # 1h RSI(14) for momentum
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:i+1])
            avg_loss[i] = np.mean(loss[1:i+1])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if np.isnan(rsi[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        vol_condition = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: RSI > 55 (momentum), price above EMA50, volume, session
            if rsi[i] > 55 and close[i] > ema50_1d_aligned[i] and vol_condition and in_session:
                signals[i] = 0.20
                position = 1
            # SHORT: RSI < 45 (momentum), price below EMA50, volume, session
            elif rsi[i] < 45 and close[i] < ema50_1d_aligned[i] and vol_condition and in_session:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: RSI < 40 or price below EMA50
            if rsi[i] < 40 or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: RSI > 60 or price above EMA50
            if rsi[i] > 60 or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals