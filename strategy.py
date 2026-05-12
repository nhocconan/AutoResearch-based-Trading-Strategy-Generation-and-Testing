#!/usr/bin/env python3
name = "12h_KAMA_RSI_Trend_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # KAMA trend filter: 1d KAMA direction
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    # Calculate KAMA
    price_diff = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    er_num = np.abs(np.diff(close_1d, n=9))
    er_den = np.nancumsum(price_diff)[9:] + 1e-10
    er = np.where(er_den != 0, er_num / er_den, 0)
    er = np.concatenate([np.zeros(9), er])
    sc = (er * 0.290 + 0.064) ** 2
    kama = np.full_like(close_1d, np.nan, dtype=float)
    kama[9] = close_1d[9]
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    kama_1d_up = kama_1d > np.roll(kama_1d, 1)
    kama_1d_down = kama_1d < np.roll(kama_1d, 1)
    kama_1d_up = np.concatenate([[False], kama_1d_up[1:]])
    kama_1d_down = np.concatenate([[False], kama_1d_down[1:]])
    kama_up_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_up)
    kama_down_aligned = align_htf_to_ltf(prices, df_1d, kama_1d_down)
    
    # RSI on 12h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    # Volume confirmation: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # ensure RSI and volume MA have enough data
    
    for i in range(start_idx, n):
        # Skip if KAMA data not ready
        if np.isnan(kama_up_aligned[i]) or np.isnan(kama_down_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA up + RSI > 50 + volume confirmation
            if kama_up_aligned[i] and rsi[i] > 50 and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down + RSI < 50 + volume confirmation
            elif kama_down_aligned[i] and rsi[i] < 50 and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: KAMA turns down
            if kama_down_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: KAMA turns up
            if kama_up_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals