#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_Chop_Filter_v3
Hypothesis: Use KAMA to determine trend direction (long when KAMA rising, short when falling) on daily timeframe, filtered by RSI (avoid overbought/oversold extremes) and Choppiness Index (only trade in trending markets). Designed for 1d timeframe to reduce trade frequency and improve robustness in both bull and bear markets.
"""

name = "1d_KAMA_Direction_RSI_Chop_Filter_v3"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def kama(close, er_len=10, fast_len=2, slow_len=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).cumsum()
        volatility = np.concatenate([[0], volatility[:-1]])
        er = change / (volatility + 1e-10)
        sc = (er * (2/(fast_len+1) - 2/(slow_len+1)) + 2/(slow_len+1))**2
        kama_out = np.zeros_like(close)
        kama_out[0] = close[0]
        for i in range(1, len(close)):
            kama_out[i] = kama_out[i-1] + sc[i] * (close[i] - kama_out[i-1])
        return kama_out
    
    kama_val = kama(close, 10, 2, 30)
    kama_dir = np.diff(kama_val, prepend=0)  # rising = 1, falling = -1, flat = 0
    
    # RSI(14) for overbought/oversold filter
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        rs = avg_gain / (avg_loss + 1e-10)
        rsi_out = 100 - (100 / (1 + rs))
        return rsi_out
    
    rsi_val = rsi(close, 14)
    
    # Choppiness Index (trend detection)
    def chop(high, low, close, length=14):
        atr = np.zeros_like(close)
        tr1 = high - low
        tr2 = np.abs(high - np.roll(close, 1))
        tr3 = np.abs(low - np.roll(close, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        atr = pd.Series(tr).ewm(alpha=1/length, adjust=False, min_periods=length).mean().values
        
        max_high = pd.Series(high).rolling(length, min_periods=length).max().values
        min_low = pd.Series(low).rolling(length, min_periods=length).min().values
        
        chop_val = 100 * np.log10((atr * length) / (max_high - min_low + 1e-10)) / np.log10(length)
        return chop_val
    
    chop_val = chop(high, low, close, 14)
    
    # Get 1-week trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    ema_34_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Entry conditions
    long_entry = (
        (kama_dir > 0) & 
        (rsi_val < 70) & 
        (chop_val < 61.8) & 
        (close > ema_34_1w_aligned)
    )
    
    short_entry = (
        (kama_dir < 0) & 
        (rsi_val > 30) & 
        (chop_val < 61.8) & 
        (close < ema_34_1w_aligned)
    )
    
    # Exit conditions
    long_exit = (kama_dir <= 0) | (rsi_val >= 70) | (chop_val >= 61.8)
    short_exit = (kama_dir >= 0) | (rsi_val <= 30) | (chop_val >= 61.8)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            if long_entry[i]:
                signals[i] = 0.25
                position = 1
            elif short_entry[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            if long_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            if short_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals