#!/usr/bin/env python3
name = "1d_KAMA_Direction_1wTrend_Filter"
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
    volume = prices['volume'].values
    
    # 1d KAMA (Kaufman Adaptive Moving Average)
    # ER = Efficiency Ratio = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close - np.roll(close, 10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)  # placeholder for rolling sum
    
    # Proper rolling volatility calculation
    volatility_rolling = pd.Series(np.abs(np.diff(close, n=1))).rolling(window=10, min_periods=10).sum().values
    # Prepend first 10 values to align lengths
    volatility_rolling = np.concatenate([np.full(10, np.nan), volatility_rolling])
    
    er = np.where(volatility_rolling != 0, change / volatility_rolling, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1w trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    # 20-period EMA on weekly
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5 * 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data for KAMA and EMA
    
    for i in range(start_idx, n):
        # Skip if KAMA or EMA data not ready
        if np.isnan(kama[i]) or np.isnan(ema_20_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price above KAMA + price above weekly EMA + volume confirmation
            if (close[i] > kama[i]) and (close[i] > ema_20_1w_aligned[i]) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + price below weekly EMA + volume confirmation
            elif (close[i] < kama[i]) and (close[i] < ema_20_1w_aligned[i]) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price below KAMA or below weekly EMA
            if (close[i] < kama[i]) or (close[i] < ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price above KAMA or above weekly EMA
            if (close[i] > kama[i]) or (close[i] > ema_20_1w_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals