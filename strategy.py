#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h1d_Trend_Confirmation"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d data for trend confirmation
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4h EMA20 for trend filter
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # 1d EMA50 for higher timeframe trend
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h RSI for entry timing
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20, 50)  # RSI(14) + EMA20(4h) + EMA50(1d)
    
    for i in range(start_idx, n):
        # Skip if required data unavailable
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_4h = ema_20_4h_aligned[i]
        ema_1d = ema_50_1d_aligned[i]
        rsi_val = rsi[i]
        
        if position == 0:
            # Enter long: RSI < 30 (oversold) AND price > 4h EMA20 AND price > 1d EMA50
            if rsi_val < 30 and close[i] > ema_4h and close[i] > ema_1d:
                signals[i] = 0.20
                position = 1
            # Enter short: RSI > 70 (overbought) AND price < 4h EMA20 AND price < 1d EMA50
            elif rsi_val > 70 and close[i] < ema_4h and close[i] < ema_1d:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: RSI > 70 (overbought) OR trend weakens (price < 4h EMA20)
            if rsi_val > 70 or close[i] < ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: RSI < 30 (oversold) OR trend weakens (price > 4h EMA20)
            if rsi_val < 30 or close[i] > ema_4h:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals