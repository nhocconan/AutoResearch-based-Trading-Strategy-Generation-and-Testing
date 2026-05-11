#!/usr/bin/env python3
name = "6h_Volume_Spike_Reversion_1dTrend"
timeframe = "6h"
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
    
    # 1. Load 1d data ONCE for trend filter (close > EMA50 = uptrend)
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 2. Volume spike filter: current volume > 2.0 * 24-period EMA (on 6h data)
    vol_ema24 = pd.Series(volume).ewm(span=24, min_periods=24, adjust=False).mean().values
    volume_spike = volume > vol_ema24 * 2.0
    
    # 3. Mean reversion trigger: price deviates >1.5*ATR from 20-period SMA
    sma20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    atr14 = pd.Series(high - low).rolling(window=14, min_periods=14).mean().values
    deviation = close - sma20
    zscore = deviation / (atr14 + 1e-10)  # avoid division by zero
    reversion_long = zscore < -1.5  # price significantly below mean
    reversion_short = zscore > 1.5   # price significantly above mean
    
    # Fixed position size to minimize churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i]) or 
            np.isnan(reversion_long[i]) or np.isnan(reversion_short[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter from 1d: only trade in direction of higher timeframe trend
        uptrend = close[i] > ema50_1d_aligned[i]
        downtrend = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # Long: price significantly below mean + volume spike + 1d uptrend
            if reversion_long[i] and volume_spike[i] and uptrend:
                signals[i] = position_size
                position = 1
            # Short: price significantly above mean + volume spike + 1d downtrend
            elif reversion_short[i] and volume_spike[i] and downtrend:
                signals[i] = -position_size
                position = -1
        else:
            # Exit when price returns to mean (z-score crosses zero) OR volume condition fails
            if position == 1:
                if zscore[i] >= 0.0 or not volume_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                if zscore[i] <= 0.0 or not volume_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals