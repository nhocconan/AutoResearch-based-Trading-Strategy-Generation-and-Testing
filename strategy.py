#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator with 1d trend filter and volume confirmation
# Williams Alligator (13,8,5 SMAs) identifies trend direction and strength
# 1d EMA50 filter ensures alignment with higher timeframe trend
# Volume spike (>1.5x average) confirms conviction
# Designed to work in trending markets with low frequency to avoid fee drag
# Target: 20-40 trades/year for 12h timeframe
name = "12h_WilliamsAlligator_1dTrend_Volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volume
    df_1d = get_htf_data(prices, '1d')
    
    # 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Williams Alligator components (13,8,5 SMAs)
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values  # 13-period SMA
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values   # 8-period SMA
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values    # 5-period SMA
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or \
           np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 1d average volume
        volume_filter = vol_ma_1d_aligned[i] > 0 and volume[i] > 1.5 * vol_ma_1d_aligned[i]
        
        if position == 0:
            # Long entry: lips > teeth > jaw (bullish alignment) + price > EMA50_1d + volume
            if lips[i] > teeth[i] and teeth[i] > jaw[i] and close[i] > ema50_1d_aligned[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: jaws > teeth > lips (bearish alignment) + price < EMA50_1d + volume
            elif jaw[i] > teeth[i] and teeth[i] > lips[i] and close[i] < ema50_1d_aligned[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Alligator lines intertwine (loss of trend) OR price < EMA50_1d
            if not (lips[i] > teeth[i] and teeth[i] > jaw[i]) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Alligator lines intertwine (loss of trend) OR price > EMA50_1d
            if not (jaw[i] > teeth[i] and teeth[i] > lips[i]) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals