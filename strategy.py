#!/usr/bin/env python3
# 4H_1D_Combined_Breakout_Pullback_Volume
# Hypothesis: On 4h timeframe, enter long when price pulls back to 20-period EMA during an uptrend defined by price > 50-period EMA on 1d, with volume confirmation.
# Enter short when price rallies to 20-period EMA during a downtrend defined by price < 50-period EMA on 1d, with volume confirmation.
# Uses 1d trend filter to avoid counter-trend trades and 4h EMA for precise entries.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "4H_1D_Combined_Breakout_Pullback_Volume"
timeframe = "4h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # 1d trend: price > 50-period EMA for uptrend, price < 50-period EMA for downtrend
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_1d = close_1d > ema_50_1d
    trend_down_1d = close_1d < ema_50_1d
    
    # 4h EMA(20) for pullback entries
    ema_20_4h = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_avg * 1.5)
    
    # Align 1d indicators to 4h
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    trend_down_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_down_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(trend_up_1d_aligned[i]) or np.isnan(trend_down_1d_aligned[i]) or np.isnan(ema_20_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price pulls back to 20-period EMA during 1d uptrend + volume confirmation
            if low[i] <= ema_20_4h[i] and trend_up_1d_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price rallies to 20-period EMA during 1d downtrend + volume confirmation
            elif high[i] >= ema_20_4h[i] and trend_down_1d_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below 20-period EMA or trend changes to down
            if close[i] < ema_20_4h[i] or trend_down_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above 20-period EMA or trend changes to up
            if close[i] > ema_20_4h[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals