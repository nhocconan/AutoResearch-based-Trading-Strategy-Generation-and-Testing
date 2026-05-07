#!/usr/bin/env python3
# 4h_PriceAction_VolumeBreakout_1dTrend
# Hypothesis: Price action breakouts with volume confirmation and daily trend filter.
# Uses 4h price action (high/low breaks) with volume spike (>1.5x) and 1d EMA50 trend.
# Works in bull markets via long breakouts and bear via short breakdowns.
# Volume filter reduces false breakouts, trend filter avoids counter-trend trades.
# Target: 20-50 trades per year (~80-200 over 4 years) with position size 0.25.

name = "4h_PriceAction_VolumeBreakout_1dTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Price action: use recent swing points
    # Look for breaks of recent 10-period high/low
    high_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume ratio: current volume / 10-period average volume
    vol_ma = pd.Series(volume).rolling(window=10, min_periods=10).mean().values
    vol_ratio = np.where(vol_ma > 0, volume / vol_ma, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 10  # Need 10 periods for calculations
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Breakout conditions: price breaks above 10-period high or below 10-period low
        breakout_up = close[i] > high_10[i-1]  # Use previous bar's high to avoid look-ahead
        breakout_down = close[i] < low_10[i-1]  # Use previous bar's low
        
        # Volume confirmation: volume > 1.5x average
        volume_confirm = vol_ratio[i] > 1.5
        
        # Trend filter from 1d EMA50
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: upward breakout + volume + uptrend
            if breakout_up and volume_confirm and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: downward breakout + volume + downtrend
            elif breakout_down and volume_confirm and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks back below 10-period low or trend reversal
            if close[i] < low_10[i] or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks back above 10-period high or trend reversal
            if close[i] > high_10[i] or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals