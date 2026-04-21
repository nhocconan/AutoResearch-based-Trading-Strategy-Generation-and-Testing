#!/usr/bin/env python3
"""
1h_MACD_Trend_With_1d_TrendFilter_V1
Hypothesis: MACD histogram crossing zero with volume confirmation on 1h,
filtered by 1d EMA trend (EMA34), provides reliable entries in both bull and bear markets.
1d trend filter reduces whipsaws; volume ensures conviction. Target: 15-35 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    # Load 1d data once for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate MACD on 1h data
    close = prices['close'].values
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False, min_periods=26).mean().values
    macd_line = ema12 - ema26
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False, min_periods=9).mean().values
    macd_hist = macd_line - signal_line
    
    # Volume average for confirmation (20-period)
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if 1d trend not ready
        if np.isnan(ema_34_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # MACD zero cross conditions
        macd_cross_up = macd_hist[i-1] <= 0 and macd_hist[i] > 0
        macd_cross_down = macd_hist[i-1] >= 0 and macd_hist[i] < 0
        
        # Volume confirmation: current volume > 1.5 * 20-period average
        volume_ok = volume[i] > 1.5 * vol_ma[i] if not np.isnan(vol_ma[i]) else False
        
        # Trend filter: EMA34 direction
        trend_long = close[i] > ema_34_1d_aligned[i]
        trend_short = close[i] < ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: MACD crosses up + volume + uptrend
            if macd_cross_up and volume_ok and trend_long:
                signals[i] = 0.20
                position = 1
            # Short: MACD crosses down + volume + downtrend
            elif macd_cross_down and volume_ok and trend_short:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Long exit: MACD crosses down or trend turns bearish
            if macd_cross_down or not trend_long:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Short exit: MACD crosses up or trend turns bullish
            if macd_cross_up or not trend_short:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_MACD_Trend_With_1d_TrendFilter_V1"
timeframe = "1h"
leverage = 1.0