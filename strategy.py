#!/usr/bin/env python3
# 12h_MultiFactor_Trend_Follow
# Hypothesis: Trend following strategy for 12h timeframe using 1w EMA trend filter, 12h EMA crossovers, and volume confirmation.
# Designed to work in both bull and bear markets by capturing major trends with proper filtering to avoid whipsaws.
# Uses weekly EMA for trend bias, daily EMA for entry timing, and volume spike for confirmation.
# Targets ~15-25 trades/year to minimize fee drag while maintaining robustness.

name = "12h_MultiFactor_Trend_Follow"
timeframe = "12h"
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
    
    # 1w data for primary trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 1w EMA20 trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_1w_up = close_1w > ema20_1w
    trend_1w_down = close_1w < ema20_1w
    
    # Align 1w trend to 12h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up.astype(float))
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down.astype(float))
    
    # 1d data for entry timing
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d EMA21 for entry signals
    close_1d = df_1d['close'].values
    ema21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 1d EMA21 to 12h
    ema21_1d_aligned = align_htf_to_ltf(prices, df_1d, ema21_1d)
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure sufficient warmup for all indicators
    
    for i in range(start_idx, n):
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(ema21_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: 1w uptrend + price above 1d EMA21 + volume spike
            if (trend_1w_up_aligned[i] > 0.5 and
                close[i] > ema21_1d_aligned[i] and
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # Short: 1w downtrend + price below 1d EMA21 + volume spike
            elif (trend_1w_down_aligned[i] > 0.5 and
                  close[i] < ema21_1d_aligned[i] and
                  volume_spike[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: 1w trend turns down OR price breaks below 1d EMA21
            if (trend_1w_up_aligned[i] <= 0.5 or
                close[i] < ema21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: 1w trend turns up OR price breaks above 1d EMA21
            if (trend_1w_down_aligned[i] <= 0.5 or
                close[i] > ema21_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals