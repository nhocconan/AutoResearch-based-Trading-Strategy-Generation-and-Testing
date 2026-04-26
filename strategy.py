#!/usr/bin/env python3
"""
6h_PivotSqueeze_Breakout_12hTrend_v1
Hypothesis: Trade 6h breakouts from Bollinger squeeze on 12h with volume confirmation and 12h EMA50 trend filter. Squeeze = Bollinger Bandwidth < 20th percentile. Breakout = close outside Bollinger Bands (20,2). Uses discrete size 0.25 to limit fee drag. Works in bull/bear via trend filter and squeeze capturing low-volatility precursors to moves.
"""

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
    
    # Get 12h data for Bollinger squeeze and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Bollinger Bands (20,2) on 12h
    close_12h_series = pd.Series(close_12h)
    ma_20 = close_12h_series.rolling(window=20, min_periods=20).mean().values
    std_20 = close_12h_series.rolling(window=20, min_periods=20).std().values
    upper_bb = ma_20 + 2 * std_20
    lower_bb = ma_20 - 2 * std_20
    bandwidth = (upper_bb - lower_bb) / np.maximum(ma_20, 1e-10)
    
    # Squeeze: bandwidth < 20th percentile (lookback 50)
    bandwidth_series = pd.Series(bandwidth)
    bandwidth_percentile = bandwidth_series.rolling(window=50, min_periods=30).quantile(0.20).values
    squeeze = bandwidth < bandwidth_percentile
    
    # 12h EMA50 trend filter
    ema_50_12h = close_12h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 12h data to 6h timeframe
    squeeze_aligned = align_htf_to_ltf(prices, df_12h, squeeze)
    upper_bb_aligned = align_htf_to_ltf(prices, df_12h, upper_bb)
    lower_bb_aligned = align_htf_to_ltf(prices, df_12h, lower_bb)
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: volume > 1.5x 20-period average on 6h
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_spike = volume / np.maximum(volume_ma, 1e-10) > 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need squeeze (50), EMA50 (50), volume MA (20)
    start_idx = max(50, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(squeeze_aligned[i]) or np.isnan(upper_bb_aligned[i]) or
            np.isnan(lower_bb_aligned[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(volume_ma[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # 12h trend alignment
        trend_12h_uptrend = close[i] > ema_50_12h_aligned[i]
        trend_12h_downtrend = close[i] < ema_50_12h_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper BB + volume squeeze + 12h uptrend
            long_breakout = (close[i] > upper_bb_aligned[i]) and \
                           (close[i-1] <= upper_bb_aligned[i-1])
            long_signal = long_breakout and squeeze_aligned[i] and volume_spike[i] and trend_12h_uptrend
            
            # Short: price breaks below lower BB + volume squeeze + 12h downtrend
            short_breakout = (close[i] < lower_bb_aligned[i]) and \
                           (close[i-1] >= lower_bb_aligned[i-1])
            short_signal = short_breakout and squeeze_aligned[i] and volume_spike[i] and trend_12h_downtrend
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price touches middle BB OR 12h trend turns down
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if (close[i] < middle_bb or not trend_12h_uptrend):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price touches middle BB OR 12h trend turns up
            middle_bb = (upper_bb_aligned[i] + lower_bb_aligned[i]) / 2
            if (close[i] > middle_bb or not trend_12h_downtrend):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_PivotSqueeze_Breakout_12hTrend_v1"
timeframe = "6h"
leverage = 1.0