#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Ichimoku_TenkanKijun_Cross_1dTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 26:
        return np.zeros(n)
    
    # 1d EMA26 trend filter
    close_1d = df_1d['close'].values
    ema26_1d = pd.Series(close_1d).ewm(span=26, adjust=False, min_periods=26).mean().values
    trend_1d = (close_1d > ema26_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d volume average for spike detection
    vol_ma20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_ma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma20_1d)
    
    # Ichimoku Cloud components (Tenkan-sen and Kijun-sen) on 12h data
    # Tenkan-sen: (9-period high + 9-period low) / 2
    high_9 = pd.Series(high).rolling(window=9, min_periods=9).max().values
    low_9 = pd.Series(low).rolling(window=9, min_periods=9).min().values
    tenkan = (high_9 + low_9) / 2.0
    
    # Kijun-sen: (26-period high + 26-period low) / 2
    high_26 = pd.Series(high).rolling(window=26, min_periods=26).max().values
    low_26 = pd.Series(low).rolling(window=26, min_periods=26).min().values
    kijun = (high_26 + low_26) / 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 26  # warmup for Ichimoku
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(tenkan[i]) or np.isnan(kijun[i]) or 
            np.isnan(trend_1d_aligned[i]) or np.isnan(vol_ma20_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 * 20-period 1d average
        vol_spike = volume[i] > (vol_ma20_1d_aligned[i] * 2.0)
        
        if position == 0:
            # Long entry: Tenkan crosses above Kijun with volume spike and 1d uptrend
            if (tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1] and 
                vol_spike and trend_1d_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short entry: Tenkan crosses below Kijun with volume spike and 1d downtrend
            elif (tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1] and 
                  vol_spike and trend_1d_aligned[i] < 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Tenkan crosses below Kijun
            if tenkan[i] < kijun[i] and tenkan[i-1] >= kijun[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Tenkan crosses above Kijun
            if tenkan[i] > kijun[i] and tenkan[i-1] <= kijun[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Ichimoku Tenkan/Kijun cross on 12h with 1d trend filter and volume confirmation.
# Works in bull markets (trend following) and bear markets (counter-trend bounces at cloud).
# Tenkan/Kijun cross provides clear entry/exit signals with low latency.
# Volume spike (2x 20-period 1d average) confirms institutional participation.
# 1d EMA26 trend filter ensures alignment with higher timeframe momentum.
# Target: 20-40 trades/year to minimize fee drag while capturing significant moves.