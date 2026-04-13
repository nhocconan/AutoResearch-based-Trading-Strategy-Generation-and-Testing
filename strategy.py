#!/usr/bin/env python3
"""
4h_12h_KAMA_Direction_With_Volume_Confirmation
Hypothesis: 4h KAMA (2,30) direction combined with 12h trend filter and daily volume > 2.0x 20-period average.
Long when KAMA trending up (current > previous) + volume condition + 12h price > EMA50.
Short when KAMA trending down (current < previous) + volume condition + 12h price < EMA50.
Exit when KAMA reverses direction.
Designed for 4h timeframe to balance trade frequency and edge in both bull and bear markets.
Target: 20-50 trades/year per symbol.
"""

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
    
    # Daily data for volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 1d volume and its 20-period moving average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean()
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20.values)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    # 12h EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # KAMA (2,30) calculation
    # ER (Efficiency Ratio) = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(np.diff(close, n=10))  # |close - close[10]|
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=1)  # sum of absolute changes
    
    # Pad arrays to match length
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    
    er = np.divide(change_padded, volatility_padded, out=np.full_like(change_padded, np.nan), where=volatility_padded!=0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA direction (current > previous = up, current < previous = down)
    kama_dir = np.diff(kama, prepend=kama[0])
    kama_up = kama_dir > 0
    kama_down = kama_dir < 0
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    position_size = 0.25  # 25% position size
    
    for i in range(50, n):
        # Skip if any required data is not ready
        if (np.isnan(kama[i]) or np.isnan(vol_ma_20_aligned[i]) or
            np.isnan(vol_1d_aligned[i]) or np.isnan(ema_50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume condition: current 1d volume > 2.0x 20-period average
        vol_condition = vol_1d_aligned[i] > (vol_ma_20_aligned[i] * 2.0)
        
        # Trend filter: only long when price > 12h EMA50, short when price < 12h EMA50
        long_trend = close[i] > ema_50_12h_aligned[i]
        short_trend = close[i] < ema_50_12h_aligned[i]
        
        # KAMA direction signals
        kama_long = kama_up[i]
        kama_short = kama_down[i]
        
        if position == 0:
            if kama_long and vol_condition and long_trend:
                position = 1
                signals[i] = position_size
            elif kama_short and vol_condition and short_trend:
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit when KAMA turns down
            if kama_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit when KAMA turns up
            if kama_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_12h_KAMA_Direction_With_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0