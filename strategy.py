#!/usr/bin/env python3
name = "6h_RSI_River_1dTrend_VolumeSpike"
timeframe = "6h"
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
    
    # === 1D DATA FOR RSI (14) ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    gain_ema = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    loss_ema = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = gain_ema / (loss_ema + 1e-10)
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_6h = align_htf_to_ltf(prices, df_1d, rsi_14)
    
    # === 1D DATA FOR EMA34 (TREND FILTER) ===
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_6h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # === VOLUME SPIKE (20-period) ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi_14_6h[i]) or 
            np.isnan(ema34_1d_6h[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: RSI < 30 (OVERSOLD) + ABOVE 1D EMA34 + VOLUME SPIKE
            if (rsi_14_6h[i] < 30 and 
                close[i] > ema34_1d_6h[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: RSI > 70 (OVERBOUGHT) + BELOW 1D EMA34 + VOLUME SPIKE
            elif (rsi_14_6h[i] > 70 and 
                  close[i] < ema34_1d_6h[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # EXIT LONG: RSI > 50 (MOMENTUM FADE) OR BELOW 1D EMA34
            if rsi_14_6h[i] > 50 or close[i] < ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: RSI < 50 (MOMENTUM FADE) OR ABOVE 1D EMA34
            if rsi_14_6h[i] < 50 or close[i] > ema34_1d_6h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals