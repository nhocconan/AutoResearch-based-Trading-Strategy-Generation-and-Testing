#!/usr/bin/env python3
name = "4h_KC20_Trend_RSI14_Exit"
timeframe = "4h"
leverage = 1.0

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
    
    # Load 4h and 1d data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 5:
        return np.zeros(n)
    
    # 4h Keltner Channel (20, 1.5) for trend and entry
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = 0
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    kc_upper = ema20 + 1.5 * atr
    kc_lower = ema20 - 1.5 * atr
    
    # 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 4h RSI(14) for exit
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(span=14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(span=14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 50)  # Wait for KC and EMA50
    
    for i in range(start_idx, n):
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(rsi[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close above KC upper in 1d uptrend
            if close[i] > kc_upper[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Close below KC lower in 1d downtrend
            elif close[i] < kc_lower[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI > 70 (overbought) or close below KC lower
            if rsi[i] > 70 or close[i] < kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI < 30 (oversold) or close above KC upper
            if rsi[i] < 30 or close[i] > kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals