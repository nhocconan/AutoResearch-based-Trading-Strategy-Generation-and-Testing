#!/usr/bin/env python3
name = "4h_RSI20_1dClose_Backtest"
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
    
    # ===== 1d Trend Filter (HTF) =====
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d Close (not EMA, simpler)
    close_1d_arr = close_1d
    
    # Align 1d close to 4h
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d_arr)
    
    # ===== RSI(20) on 4h =====
    delta = np.diff(close, prepend=close[0])
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing (alpha = 1/period)
    roll_up = pd.Series(up).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    roll_down = pd.Series(down).ewm(alpha=1/20, adjust=False, min_periods=20).mean().values
    
    rs = roll_up / (roll_down + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # ===== Volume Spike Filter =====
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # RSI needs 20, volume avg needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or 
            np.isnan(close_1d_aligned[i]) or
            np.isnan(vol_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: RSI < 20 (oversold) + price above 1d close + volume spike
            if (rsi[i] < 20 and 
                close[i] > close_1d_aligned[i] and
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: RSI > 80 (overbought) + price below 1d close + volume spike
            elif (rsi[i] > 80 and 
                  close[i] < close_1d_aligned[i] and
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: RSI > 60 OR price below 1d close
            if rsi[i] > 60 or close[i] < close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: RSI < 40 OR price above 1d close
            if rsi[i] < 40 or close[i] > close_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals