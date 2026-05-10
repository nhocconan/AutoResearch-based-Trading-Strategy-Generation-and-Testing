#!/usr/bin/env python3
# 4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) act as key support/resistance in intraday trading. Breakouts beyond these levels with volume confirmation and daily EMA trend filter capture strong moves in both bull and bear markets. Using 1d EMA34 as trend filter reduces whipsaws by ensuring trades align with higher timeframe momentum. Low trade frequency expected due to strict breakout conditions + volume confirmation + trend filter.

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_Trend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Previous day's Camarilla levels (using prior 1d close)
    prev_close_1d = np.roll(close_1d, 1)
    prev_close_1d[0] = np.nan  # First value invalid
    
    # Camarilla R1 and S1 levels
    r1 = prev_close_1d + (1.1/12) * (high_1d := np.roll(df_1d['high'].values, 1)) - (1.1/12) * (low_1d := np.roll(df_1d['low'].values, 1))
    s1 = prev_close_1d - (1.1/12) * (high_1d) + (1.1/12) * (low_1d)
    
    # Align Camarilla levels to 4h timeframe (wait for 1d bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h ATR for volatility filter
    tr1 = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr1[0] = np.nan
    atr = pd.Series(tr1).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Volume confirmation (20-period average)
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Need enough history for EMA and calculations
    
    for i in range(start_idx, n):
        if np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or \
           np.isnan(ema_34_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above daily EMA34, volume confirmation
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below daily EMA34, volume confirmation
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i] and volume[i] > 1.5 * vol_ma[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price drops below S1 OR below daily EMA34
            if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price rises above R1 OR above daily EMA34
            if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals