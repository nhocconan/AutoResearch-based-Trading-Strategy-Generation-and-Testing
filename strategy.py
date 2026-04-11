#!/usr/bin/env python3
# 4h_1d_ema_crossover_volume
# Strategy: 4-hour EMA crossover with 1-day trend filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Capture momentum in direction of daily trend using 4h EMA crossovers,
# with volume confirmation to avoid false signals. Works in bull and bear by
# following the dominant daily trend. Uses discrete position sizing to minimize
# fee churn.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_ema_crossover_volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load higher timeframe data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h data for EMA calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # 1d EMA for trend filter
    close_1d = df_1d['close'].values
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d EMAs to 4h timeframe (wait for daily close)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 4h EMA for crossover signal
    close_4h = df_4h['close'].values
    ema_9_4h = pd.Series(close_4h).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21_4h = pd.Series(close_4h).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Align 4h EMAs to 4h timeframe (no additional delay needed)
    ema_9_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_9_4h)
    ema_21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_21_4h)
    
    # Volume average (20-period) for confirmation
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_avg)  # Volume spike filter
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(ema_9_4h_aligned[i]) or np.isnan(ema_21_4h_aligned[i]) or
            np.isnan(vol_avg[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: price above/below 1d EMA50
        uptrend_1d = ema_20_1d_aligned[i] > ema_50_1d_aligned[i]
        downtrend_1d = ema_20_1d_aligned[i] < ema_50_1d_aligned[i]
        
        # EMA crossover signals
        bullish_cross = ema_9_4h_aligned[i] > ema_21_4h_aligned[i]
        bearish_cross = ema_9_4h_aligned[i] < ema_21_4h_aligned[i]
        
        # Entry conditions: crossover in direction of daily trend with volume confirmation
        long_signal = bullish_cross and uptrend_1d and vol_spike[i]
        short_signal = bearish_cross and downtrend_1d and vol_spike[i]
        
        # Exit conditions: opposite crossover
        exit_long = position == 1 and bearish_cross
        exit_short = position == -1 and bullish_cross
        
        # Trading logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and exit_long:
            position = 0
            signals[i] = 0.0
        elif position == -1 and exit_short:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals