#!/usr/bin/env python3
name = "6h_AroonTrend_1dPullback"
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
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d Aroon indicators for trend direction
    high_series = pd.Series(df_1d['high'])
    low_series = pd.Series(df_1d['low'])
    aroon_up = high_series.rolling(window=25, min_periods=25).apply(
        lambda x: 100 * (24 - x.argmax()) / 24, raw=False
    ).values
    aroon_down = low_series.rolling(window=25, min_periods=25).apply(
        lambda x: 100 * (24 - x.argmin()) / 24, raw=False
    ).values
    aroon_up_1d = align_htf_to_ltf(prices, df_1d, aroon_up)
    aroon_down_1d = align_htf_to_ltf(prices, df_1d, aroon_down)
    
    # 6h EMA(20) for pullback entry
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Wait for Aroon and EMA
    
    for i in range(start_idx, n):
        if np.isnan(aroon_up_1d[i]) or np.isnan(aroon_down_1d[i]) or np.isnan(ema_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1d uptrend (Aroon Up > 70) + price pulls back to EMA(20)
            if aroon_up_1d[i] > 70 and close[i] <= ema_20[i] * 1.005 and close[i] >= ema_20[i] * 0.995:
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend (Aroon Down > 70) + price pulls back to EMA(20)
            elif aroon_down_1d[i] > 70 and close[i] <= ema_20[i] * 1.005 and close[i] >= ema_20[i] * 0.995:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: 1d trend reversal (Aroon Down > 50) or strong adverse move
            if aroon_down_1d[i] > 50 or close[i] < ema_20[i] * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: 1d trend reversal (Aroon Up > 50) or strong adverse move
            if aroon_up_1d[i] > 50 or close[i] > ema_20[i] * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6s Aroon trend filter with 6h EMA pullback entry
# - Aroon(25) on 1d identifies strong trends (>70) and weakening trends (<50)
# - Enter on 6h pullbacks to EMA(20) during strong 1d trends
# - Works in bull (buy pullbacks in uptrend) and bear (sell pullbacks in downtrend)
# - Aroon exit provides timely trend reversal signals
# - Position size 0.25 targets 15-25 trades/year, avoiding fee drag
# - Pullback entry improves risk-reward vs breakout chasing