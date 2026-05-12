#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_12hTrend_VolumeS"
timeframe = "4h"
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
    
    # 12h trend: EMA50
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Camarilla pivot levels from 1d
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC (already complete due to get_ftf_data)
    ph = high_1d  # previous day high
    pl = low_1d   # previous day low
    pc = close_1d # previous day close
    
    # Camarilla levels
    R1 = pc + (ph - pl) * 1.1 / 12
    S1 = pc - (ph - pl) * 1.1 / 12
    
    # Align to 4h
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # ensure sufficient data for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema_12h_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price > R1 + 12h uptrend + volume confirmation
            if (close[i] > R1_aligned[i]) and (close[i] > ema_12h_aligned[i]) and vol_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price < S1 + 12h downtrend + volume confirmation
            elif (close[i] < S1_aligned[i]) and (close[i] < ema_12h_aligned[i]) and vol_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Price < S1 or 12h downtrend
            if (close[i] < S1_aligned[i]) or (close[i] < ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Price > R1 or 12h uptrend
            if (close[i] > R1_aligned[i]) or (close[i] > ema_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals