#!/usr/bin/env python3
# 4h_Donchian_Breakout_Volume_12hTrend
# Hypothesis: Combines 4h Donchian breakouts with 12h EMA trend filter and volume confirmation
# to capture strong momentum moves while avoiding counter-trend trades. Designed for low trade
# frequency (20-40/year) to minimize fee drag and perform well in both bull and bear markets.

name = "4h_Donchian_Breakout_Volume_12hTrend"
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
    
    # 4h Donchian Channel (20-period)
    donch_period = 20
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donch_high = high_series.rolling(window=donch_period, min_periods=donch_period).max()
    donch_low = low_series.rolling(window=donch_period, min_periods=donch_period).min()
    
    # 4h ATR for volume-normalized volatility (used in volume filter)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean()
    
    # 12h EMA trend (HTF)
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 1.5 * 20-period average
    def mean_arr(arr, p):
        res = np.full_like(arr, np.nan)
        if len(arr) >= p:
            for i in range(p-1, len(arr)):
                res[i] = np.mean(arr[i-p+1:i+1])
        return res
    vol_ma = mean_arr(volume, 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        if np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or \
           np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_confirm = volume[i] > 1.5 * vol_ma[i] if vol_ma[i] > 0 else False
        
        if position == 0:
            # Long: price breaks above Donchian high, above 12h EMA trend, with volume
            if close[i] > donch_high[i] and close[i] > ema_12h_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low, below 12h EMA trend, with volume
            elif close[i] < donch_low[i] and close[i] < ema_12h_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low or trend changes
            if close[i] < donch_low[i] or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high or trend changes
            if close[i] > donch_high[i] or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals