#!/usr/bin/env python3
# 4h_SMA_Crossover_Volume_Trend
# Hypothesis: In trending markets, when the 4h price crosses above/below a 50-period SMA with volume confirmation, it signals momentum continuation. The trend filter (1d EMA50) ensures alignment with higher timeframe direction, reducing false signals in sideways markets. Designed for low trade frequency to avoid fee drag.

name = "4h_SMA_Crossover_Volume_Trend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 50-period SMA on 4h close
    sma_50 = pd.Series(close).rolling(window=50, min_periods=50).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        if position == 0:
            # LONG: Price crosses above SMA50 with volume confirmation in uptrend (close > EMA50_1d)
            if not np.isnan(sma_50[i]) and not np.isnan(ema_50_1d_aligned[i]) and \
               close[i] > sma_50[i] and close[i-1] <= sma_50[i-1] and \
               volume_confirmed[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price crosses below SMA50 with volume confirmation in downtrend (close < EMA50_1d)
            elif not np.isnan(sma_50[i]) and not np.isnan(ema_50_1d_aligned[i]) and \
                 close[i] < sma_50[i] and close[i-1] >= sma_50[i-1] and \
                 volume_confirmed[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below SMA50 or trend weakens (close < EMA50_1d)
            if not np.isnan(sma_50[i]) and not np.isnan(ema_50_1d_aligned[i]) and \
               (close[i] < sma_50[i] and close[i-1] >= sma_50[i-1]) or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above SMA50 or trend weakens (close > EMA50_1d)
            if not np.isnan(sma_50[i]) and not np.isnan(ema_50_1d_aligned[i]) and \
               (close[i] > sma_50[i] and close[i-1] <= sma_50[i-1]) or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals