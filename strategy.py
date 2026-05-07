#!/usr/bin/env python3
# 1h_VolatilityRegime_CamarillaBreakout_4hTrend
# Hypothesis: Use 4h trend (close vs EMA34) for direction, 1h for entry timing. 
# Entry when price breaks 1d Camarilla R1/S1 with volume spike, only in high volatility regime (ATR ratio > 1.2).
# Volatility regime filter reduces whipsaws in ranging markets. Target 15-30 trades/year.

name = "1h_VolatilityRegime_CamarillaBreakout_4hTrend"
timeframe = "1h"
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
    
    # 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 35:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # 1d data for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    camarilla_range = high_1d - low_1d
    r1 = close_1d + 1.1 * camarilla_range / 12
    s1 = close_1d - 1.1 * camarilla_range / 12
    r1_1h = align_htf_to_ltf(prices, df_1d, r1)
    s1_1h = align_htf_to_ltf(prices, df_1d, s1)
    
    # 1h volatility regime: ATR ratio (current ATR / 50-period ATR)
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr1[0] = high[0] - low[0]
    atr_14 = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    atr_50 = pd.Series(tr1).rolling(window=50, min_periods=50).mean().values
    atr_ratio = atr_14 / atr_50
    high_vol = atr_ratio > 1.2  # volatile regime
    
    # Volume spike (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        if (np.isnan(ema_34_4h_aligned[i]) or np.isnan(r1_1h[i]) or np.isnan(s1_1h[i]) or
            np.isnan(volume_spike[i]) or np.isnan(high_vol[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > R1, above 4h EMA34 trend, volume spike, high volatility
            if close[i] > r1_1h[i] and close[i] > ema_34_4h_aligned[i] and volume_spike[i] and high_vol[i]:
                signals[i] = 0.20
                position = 1
            # Short: price < S1, below 4h EMA34 trend, volume spike, high volatility
            elif close[i] < s1_1h[i] and close[i] < ema_34_4h_aligned[i] and volume_spike[i] and high_vol[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price < R1 or below 4h EMA34 trend
            if close[i] < r1_1h[i] or close[i] < ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price > S1 or above 4h EMA34 trend
            if close[i] > s1_1h[i] or close[i] > ema_34_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals