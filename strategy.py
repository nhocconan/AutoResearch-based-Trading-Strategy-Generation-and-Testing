#!/usr/bin/env python3
# 1H_Pullback_TO_Trend_EMA21_4hTrend
# Hypothesis: On 1h timeframe, enter pullbacks to EMA21 in direction of 4h EMA50 trend.
# Uses 4h EMA50 for trend filter (higher timeframe direction) and 1h EMA21 for pullback entries.
# Reduces false signals by trading with higher timeframe trend, targets 15-30 trades/year.
# Works in bull/bear markets by aligning with 4h trend and using pullback entries for better risk/reward.

name = "1H_Pullback_TO_Trend_EMA21_4hTrend"
timeframe = "1h"
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
    
    # Get 4h data for EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend direction
    ema_50_4h = pd.Series(df_4h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate 1h EMA(21) for pullback entries
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 20)  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_21[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: price above/below 4h EMA50
        price_above_4h_ema = close[i] > ema_50_4h_aligned[i]
        price_below_4h_ema = close[i] < ema_50_4h_aligned[i]
        
        if position == 0:
            # Long entry: pullback to 1h EMA21 + above 4h EMA50 + volume filter
            if (close[i] > ema_21[i] and 
                price_above_4h_ema and 
                volume[i] > vol_threshold[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: pullback to 1h EMA21 + below 4h EMA50 + volume filter
            elif (close[i] < ema_21[i] and 
                  price_below_4h_ema and 
                  volume[i] > vol_threshold[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below 1h EMA21 or trend changes
            if (close[i] < ema_21[i] or not price_above_4h_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above 1h EMA21 or trend changes
            if (close[i] > ema_21[i] or not price_below_4h_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals