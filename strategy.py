#!/usr/bin/env python3
# 12h_1d_ema_rsi_volume_breakout_v1
# Hypothesis: 12-hour EMA trend with RSI pullback and volume confirmation on daily chart.
# Long: price > 12h EMA200 AND RSI(14) < 40 AND daily volume > 1.5x 20-day average volume.
# Short: price < 12h EMA200 AND RSI(14) > 60 AND daily volume > 1.5x 20-day average volume.
# Exit: price crosses back over 12h EMA200.
# Designed to capture trend continuation after pullbacks in both bull and bear markets with strict entry criteria.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_ema_rsi_volume_breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h EMA200 for trend filter
    ema_200 = np.full(n, np.nan)
    if n >= 200:
        ema_200[199] = np.mean(close[:200])
        for i in range(200, n):
            ema_200[i] = close[i] * (2/201) + ema_200[i-1] * (199/201)
    
    # 14-period RSI
    rsi = np.full(n, np.nan)
    if n >= 14:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(n, np.nan)
        avg_loss = np.full(n, np.nan)
        avg_gain[13] = np.mean(gain[1:14])
        avg_loss[13] = np.mean(loss[1:14])
        for i in range(14, n):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi[i] = 100 - (100 / (1 + rs))
    
    # Daily volume average for confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    avg_vol_1d = np.full(len(vol_1d), np.nan)
    for i in range(20, len(vol_1d)):
        avg_vol_1d[i] = np.mean(vol_1d[i-20:i])
    
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        price = close[i]
        ema_val = ema_200[i]
        rsi_val = rsi[i]
        vol_surge = volume[i] > 1.5 * avg_vol_1d_aligned[i] if not np.isnan(avg_vol_1d_aligned[i]) else False
        
        if np.isnan(ema_val) or np.isnan(rsi_val):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            if price < ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            if price > ema_val:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if price > ema_val and rsi_val < 40 and vol_surge:
                position = 1
                signals[i] = 0.25
            elif price < ema_val and rsi_val > 60 and vol_surge:
                position = -1
                signals[i] = -0.25
    
    return signals