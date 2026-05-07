#!/usr/bin/env python3
# 12h_KAMA_1dTrend_Volume
# Hypothesis: Uses Kaufman Adaptive Moving Average (KAMA) on 12h to determine trend direction,
# filtered by 1d EMA34 trend and volume spike. KAMA adapts to market noise, reducing false signals
# in choppy markets while capturing strong trends. Works in both bull and bear markets by only
# trading in the direction of the higher timeframe trend. Target: 15-30 trades/year to stay within
# optimal frequency range and minimize fee drag.

name = "12h_KAMA_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate KAMA on 12h data
    # KAMA parameters: ER lookback = 10, fast EMA = 2, slow EMA = 30
    er_period = 10
    fast_ema = 2
    slow_ema = 30
    
    change = np.abs(np.diff(close, n=er_period))
    volatility = np.sum(np.abs(np.diff(close)), axis=1)
    # Avoid division by zero
    er = np.divide(change, volatility, out=np.zeros_like(change, dtype=float), where=volatility!=0)
    sc = np.power(er * (2/(fast_ema+1) - 2/(slow_ema+1)) + 2/(slow_ema+1), 2)
    
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Calculate volume spike on 12h timeframe (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(kama[i]) or np.isnan(ema_34_1d_12h[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA + above 1d EMA34 + volume spike
            if close[i] > kama[i] and close[i] > ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA + below 1d EMA34 + volume spike
            elif close[i] < kama[i] and close[i] < ema_34_1d_12h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price crosses below KAMA or below 1d EMA34
            if close[i] < kama[i] or close[i] < ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price crosses above KAMA or above 1d EMA34
            if close[i] > kama[i] or close[i] > ema_34_1d_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals