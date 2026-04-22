#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    # Hypothesis: Weekly EMA200 trend + Daily KAMA direction + 12h volume confirmation
    # Weekly trend filter prevents counter-trend trades in strong trends
    # Daily KAMA adapts to volatility for trend detection
    # 12h volume surge confirms institutional participation
    # Works in bull markets (buy pullbacks in uptrend) and bear markets (sell rallies in downtrend)
    
    # Load weekly data once
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Weekly EMA200 trend filter
    ema_1w_200 = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_1w_200_aligned = align_htf_to_ltf(prices, df_1w, ema_1w_200)
    
    # Load daily data once
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Daily KAMA (adaptive moving average)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d, 2))  # 2-period volatility
    volatility = np.concatenate([[volatility[0]], volatility])  # align lengths
    
    # Efficiency ratio
    er = np.zeros_like(close_1d)
    er[1:] = np.abs(np.diff(close_1d))[1:] / (volatility[1:] + 1e-10)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 12h volume confirmation (align volume data)
    df_12h = get_htf_data(prices, '12h')
    vol_12h = df_12h['volume'].values
    vol_ma20_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma20_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma20_12h)
    vol_surge = df_12h['volume'].values > 1.5 * vol_ma20_12h
    vol_surge_aligned = align_htf_to_ltf(prices, df_12h, vol_surge)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(200, n):
        # Skip if data not ready
        if (np.isnan(ema_1w_200_aligned[i]) or np.isnan(kama_aligned[i]) or
            np.isnan(vol_ma20_12h_aligned[i]) or np.isnan(vol_surge_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA200 AND above daily KAMA with volume surge
            if close[i] > ema_1w_200_aligned[i] and close[i] > kama_aligned[i] and vol_surge_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA200 AND below daily KAMA with volume surge
            elif close[i] < ema_1w_200_aligned[i] and close[i] < kama_aligned[i] and vol_surge_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses daily KAMA (trend change signal)
            if position == 1:
                if close[i] < kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > kama_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12h_WeeklyEMA200_DailyKAMA_12hVolumeSurge_v1"
timeframe = "12h"
leverage = 1.0