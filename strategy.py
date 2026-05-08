#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12-hour data once for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 trend filter
    close_12h = df_12h['close'].values
    ema20_12h = pd.Series(close_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    trend_12h = (close_12h > ema20_12h).astype(float)
    trend_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_12h)
    
    # Get daily data for Donchian channel and volume calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian channel (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    high_20d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    high_20d_aligned = align_htf_to_ltf(prices, df_1d, high_20d)
    low_20d_aligned = align_htf_to_ltf(prices, df_1d, low_20d)
    
    # Daily volume spike detection: current volume > 1.5 * 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma20d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma20d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20d)
    vol_spike = volume > (vol_ma20d_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20d_aligned[i]) or np.isnan(low_20d_aligned[i]) or 
            np.isnan(trend_12h_aligned[i]) or np.isnan(vol_ma20d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: price breaks above 20-day high with volume spike and 12h uptrend
            long_cond = (close[i] > high_20d_aligned[i] and vol_spike[i] and trend_12h_aligned[i] > 0.5)
            
            # Short entry: price breaks below 20-day low with volume spike and 12h downtrend
            short_cond = (close[i] < low_20d_aligned[i] and vol_spike[i] and trend_12h_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price closes below 20-day low (mean reversion)
            if close[i] < low_20d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above 20-day high (mean reversion)
            if close[i] > high_20d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on daily timeframe with volume confirmation and 12H trend filter on 4H timeframe.
# Works in bull markets (breakouts continue) and bear markets (mean reversion at opposite band).
# 12h EMA20 ensures alignment with intermediate-term trend, reducing counter-trend trades.
# Volume spike filter (1.5x 20-day average) ensures momentum confirmation.
# Target: 20-30 trades/year to minimize fee decay while capturing significant moves.
# Uses 4H timeframe for execution with daily/12h HTF filters to avoid overtrading.