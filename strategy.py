#!/usr/bin/env python3
# 1h_Bollinger_MeanReversion_4hTrend_VolumeFilter
# Hypothesis: Use Bollinger Band mean reversion with 4h trend filter and volume spike.
# Long when price touches lower Bollinger band with 4h uptrend and volume > 1.5x MA.
# Short when price touches upper Bollinger band with 4h downtrend and volume > 1.5x MA.
# Exit when price returns to middle Bollinger band.
# Uses 4h for trend direction (avoiding whipsaw) and 1h for precise entry timing.
# Designed to work in both bull and bear markets by filtering with 4h trend.
# Targets 15-30 trades/year to minimize fee drag on 1h timeframe.

name = "1h_Bollinger_MeanReversion_4hTrend_VolumeFilter"
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
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Bollinger Bands (20-period, 2 std dev)
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    middle = sma20
    
    # 4h trend filter: EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    ema50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(middle[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price touches lower band with 4h uptrend and volume spike
            if (low[i] <= lower[i] and 
                ema50_4h_aligned[i] > ema50_4h_aligned[i-1] and  # 4h uptrend confirmation
                volume[i] > vol_ma[i] * 1.5):
                signals[i] = 0.20
                position = 1
            # SHORT: Price touches upper band with 4h downtrend and volume spike
            elif (high[i] >= upper[i] and 
                  ema50_4h_aligned[i] < ema50_4h_aligned[i-1] and  # 4h downtrend confirmation
                  volume[i] > vol_ma[i] * 1.5):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price returns to middle band
            if close[i] >= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price returns to middle band
            if close[i] <= middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals