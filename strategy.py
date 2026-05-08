#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_KAMA_Trend_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate KAMA on daily close
    # KAMA parameters: ER period=10, Fast=2, Slow=30
    change = np.abs(np.diff(close))
    change = np.concatenate([[0], change])
    volatility = np.abs(np.diff(close))
    volatility = np.concatenate([[0], volatility])
    
    # Efficiency Ratio
    er_num = np.abs(np.subtract(close[9:], close[:-9]))  # 10-period change
    er_den = np.sum(np.lib.stride_tricks.sliding_window_view(volatility, 10), axis=1)
    er = np.concatenate([np.full(9, np.nan), np.divide(er_num, er_den, out=np.zeros_like(er_den), where=er_den!=0)])
    # Smoothing constants
    sc = np.power(er * (2/2 - 2/30) + 2/30, 2)
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[9] = close[9]  # seed
    for i in range(10, n):
        if not np.isnan(kama[i-1]) and not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Weekly trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    # Weekly EMA21 for trend
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    # Volume confirmation: 10-day average volume
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = volume / np.where(vol_ma > 0, vol_ma, 1.0)
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 50  # enough for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_21_1w_aligned[i]) or 
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price > KAMA and above weekly EMA21 with volume
            if (close[i] > kama[i] and 
                close[i] > ema_21_1w_aligned[i] and
                vol_ratio[i] > 1.3):
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA and below weekly EMA21 with volume
            elif (close[i] < kama[i] and 
                  close[i] < ema_21_1w_aligned[i] and
                  vol_ratio[i] > 1.3):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals