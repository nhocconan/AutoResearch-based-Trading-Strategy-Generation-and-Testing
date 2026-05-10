#!/usr/bin/env python3
# 1h_RSI_4hTrend_1dVolume
# Hypothesis: Uses RSI(14) mean reversion on 1h with 4h trend filter and 1d volume confirmation.
# Long when RSI<30 in 4h uptrend with 1d volume>1.5x average.
# Short when RSI>70 in 4h downtrend with 1d volume>1.5x average.
# Designed for low trade frequency (~20-50/year) to avoid fee drag, works in bull/bear via trend filter.

name = "1h_RSI_4hTrend_1dVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate RSI on 1h close
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate 4h EMA20 for trend
    ema_20_4h = pd.Series(df_4h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d volume average
    vol_avg_1d = pd.Series(df_1d['volume']).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, 20)  # Warmup for RSI and EMA
    
    for i in range(start_idx, n):
        if np.isnan(rsi[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_avg_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 4h
        uptrend = close[i] > ema_20_4h_aligned[i]
        downtrend = close[i] < ema_20_4h_aligned[i]
        
        # Volume confirmation from 1d
        volume_confirm = volume[i] > vol_avg_1d_aligned[i] * 1.5
        
        if position == 0:
            # Long entry: RSI oversold in uptrend with volume confirmation
            if rsi[i] < 30 and uptrend and volume_confirm:
                signals[i] = 0.20
                position = 1
            # Short entry: RSI overbought in downtrend with volume confirmation
            elif rsi[i] > 70 and downtrend and volume_confirm:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: RSI overbought or trend ends
            if rsi[i] > 70 or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: RSI oversold or trend ends
            if rsi[i] < 30 or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals