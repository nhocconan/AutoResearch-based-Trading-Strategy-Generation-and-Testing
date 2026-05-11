#!/usr/bin/env python3
name = "1d_1w_KAMA_Trend_RSI_Pullback"
timeframe = "1d"
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
    
    # 1d KAMA trend
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.abs(np.diff(close))
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    kama_smooth = pd.Series(kama).ewm(span=5, adjust=False, min_periods=5).mean().values
    kama_dir = kama_smooth > np.roll(kama_smooth, 1)
    kama_dir[0] = False
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # 1d RSI pullback
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.divide(avg_gain, avg_loss, out=np.zeros_like(avg_gain), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_ok = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema_20_1w_aligned[i]) or np.isnan(rsi[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Uptrend (price > weekly EMA20), KAMA up, RSI pullback (<40), volume
            if close[i] > ema_20_1w_aligned[i] and kama_dir[i] and rsi[i] < 40 and volume_ok[i]:
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < weekly EMA20), KAMA down, RSI bounce (>60), volume
            elif close[i] < ema_20_1w_aligned[i] and not kama_dir[i] and rsi[i] > 60 and volume_ok[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down or RSI > 70
            if not kama_dir[i] or rsi[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up or RSI < 30
            if kama_dir[i] or rsi[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals