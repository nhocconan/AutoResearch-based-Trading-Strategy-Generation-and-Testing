#!/usr/bin/env python3
name = "6h_Three_Pillar_Signal"
timeframe = "6h"
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
    
    # 1. 12h EMA(21) for trend filter - updated to shorter period for responsiveness
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    ema_12h = pd.Series(df_12h['close']).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 2. 24h RSI(14) for overbought/oversold - uses 24h data (two 12h candles)
    df_24h = get_htf_data(prices, '12h')
    if len(df_24h) < 28:  # Need 14*2 = 28 periods for 24h RSI
        return np.zeros(n)
    close_24h = df_24h['close'].values
    delta = np.diff(close_24h, prepend=close_24h[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_24h = 100 - (100 / (1 + rs))
    # Align to 6h: each 24h bar = 4 6h bars
    rsi_24h_repeated = np.repeat(rsi_24h, 4)
    if len(rsi_24h_repeated) > n:
        rsi_24h_repeated = rsi_24h_repeated[:n]
    else:
        rsi_24h_repeated = np.pad(rsi_24h_repeated, (0, n - len(rsi_24h_repeated)), 'edge')
    
    # 3. Volume spike detection - > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        if np.isnan(ema_12h_aligned[i]) or np.isnan(rsi_24h_repeated[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Uptrend (price > EMA), RSI oversold (<30), volume spike
            if (close[i] > ema_12h_aligned[i] and 
                rsi_24h_repeated[i] < 30 and 
                vol_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: Downtrend (price < EMA), RSI overbought (>70), volume spike
            elif (close[i] < ema_12h_aligned[i] and 
                  rsi_24h_repeated[i] > 70 and 
                  vol_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: RSI overbought (>70) or trend change
            if rsi_24h_repeated[i] > 70 or close[i] < ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: RSI oversold (<30) or trend change
            if rsi_24h_repeated[i] < 30 or close[i] > ema_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Combines 12h trend filter (EMA21), 24h RSI for extreme conditions, and volume spikes.
# In trending markets, buys dips in uptrends and sells rallies in downtrends during high volume.
# Works in both bull and bear markets by following the 12h trend while using RSI for entry timing.
# Volume spike ensures institutional participation. Target: 20-40 trades/year to avoid fee drag.
# Position size 0.25 balances capture and drawdown control. 6h timeframe reduces noise vs lower TFs.
# 24h RSI uses two 12h candles for smoother readings vs 6h RSI which is too noisy.