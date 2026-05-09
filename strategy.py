#!/usr/bin/env python3
# Hypothesis: 4h RSI mean-reversion with 1d volume spike and 1w trend filter
# In overbought/oversold conditions (RSI > 70 or < 30) on 4h, expect mean reversion.
# Only take trades when 1d volume is above its 20-period average (confirming interest).
# Only take trades in direction of 1w EMA50 trend (above EMA50 = long bias, below = short bias).
# Exit when RSI returns to neutral zone (40-60).
# Target: 20-50 trades per year with size 0.25.

name = "4h_RSI_MeanRev_1dVol_1wTrend"
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
    
    # 4h RSI (14-period)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    # 1d volume and its 20-period average for volume spike filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume']
    volume_ma_20 = volume_1d.rolling(window=20, min_periods=20).mean()
    volume_spike = volume_1d > volume_ma_20
    volume_spike_values = volume_spike.values
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_values)
    
    # 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close']
    ema_50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean()
    ema_50_1w_values = ema_50_1w.values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w_values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(rsi[i]) or
            np.isnan(volume_spike_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: RSI oversold (<30) + volume spike + above 1w EMA50
            if (rsi[i] < 30 and 
                volume_spike_aligned[i] and 
                close[i] > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: RSI overbought (>70) + volume spike + below 1w EMA50
            elif (rsi[i] > 70 and 
                  volume_spike_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: RSI returns to neutral (>=40)
            if rsi[i] >= 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: RSI returns to neutral (<=60)
            if rsi[i] <= 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals