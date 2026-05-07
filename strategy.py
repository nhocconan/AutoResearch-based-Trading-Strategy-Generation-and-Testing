#!/usr/bin/env python3
name = "1d_KAMA_Direction_RSI_ChopFilter_v1"
timeframe = "1d"
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
    
    # Calculate 1-day KAMA (Kaufman Adaptive Moving Average)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close, prepend=close[0]))[:10])
    er = np.zeros(n)
    er[9:] = change[9:] / np.maximum(volatility, 1e-10)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate 14-period RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 14-period Choppiness Index
    atr = np.zeros(n)
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[0], tr])
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = np.zeros(n)
    for i in range(13, n):
        if atr[i] > 0 and (max_high[i] - min_low[i]) > 0:
            chop[i] = 100 * np.log10(np.sum(tr[i-13:i+1]) / np.log(14) / (max_high[i] - min_low[i]))
        else:
            chop[i] = 50
    
    # Load 1-week trend filter (EMA50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    trend_up = close > ema_50_1w_aligned
    trend_down = close < ema_50_1w_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_last_trade = 0
    cooldown_bars = 3  # 3 days to prevent overtrading
    
    start_idx = max(30, 14)  # Ensure enough data for KAMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama[i]) or 
            np.isnan(rsi[i]) or 
            np.isnan(chop[i]) or 
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                bars_since_last_trade += 1
            continue
        
        bars_since_last_trade += 1
        
        # Determine trend direction
        trending_up = trend_up[i]
        trending_down = trend_down[i]
        
        if position == 0 and bars_since_last_trade >= cooldown_bars:
            # Long: Price above KAMA, RSI < 30 (oversold), Chop > 61.8 (ranging) in 1w uptrend
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop[i] > 61.8 and 
                trending_up):
                signals[i] = 0.25
                position = 1
                bars_since_last_trade = 0
            # Short: Price below KAMA, RSI > 70 (overbought), Chop > 61.8 (ranging) in 1w downtrend
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop[i] > 61.8 and 
                  trending_down):
                signals[i] = -0.25
                position = -1
                bars_since_last_trade = 0
        elif position == 1:
            # Exit: Price crosses below KAMA or RSI > 70
            if close[i] < kama[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or RSI < 30
            if close[i] > kama[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
                bars_since_last_trade = 0
            else:
                signals[i] = -0.25
    
    return signals