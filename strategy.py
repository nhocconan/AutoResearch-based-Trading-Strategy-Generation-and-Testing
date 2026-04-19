#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Bollinger Bands + 1d Volume + 4h CCI momentum
# Bollinger Bands (20, 2) for volatility-based entries
# 1d volume spike (>2x average) for conviction
# 4h CCI (20) for momentum confirmation (CCI > 100 long, < -100 short)
# Exit on opposite band touch or CCI mean reversion
# Designed to work in trending markets with volume confirmation
# Target: 20-35 trades/year to avoid fee drag
name = "4h_BB_CCI_1dVolume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Volume average (20-period)
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # 4h Bollinger Bands (20, 2)
    sma = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper = sma + 2 * std
    lower = sma - 2 * std
    
    # 4h CCI (20)
    tp = (high + low + close) / 3
    sma_tp = pd.Series(tp).rolling(window=20, min_periods=20).mean().values
    mad = pd.Series(tp).rolling(window=20, min_periods=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True).values
    cci = (tp - sma_tp) / (0.015 * mad)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(sma[i]) or np.isnan(std[i]) or np.isnan(cci[i]) or np.isnan(vol_ma_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2x average
        if i >= 20:
            vol_ma = vol_ma_1d_aligned[i]
        else:
            vol_ma = vol_ma_1d_aligned[i] if not np.isnan(vol_ma_1d_aligned[i]) else volume[i]
        volume_filter = vol_ma > 0 and volume[i] > 2 * vol_ma
        
        if position == 0:
            # Long entry: price touches lower BB + CCI > 100 + volume
            if close[i] <= lower[i] and cci[i] > 100 and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short entry: price touches upper BB + CCI < -100 + volume
            elif close[i] >= upper[i] and cci[i] < -100 and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price touches upper BB OR CCI < -50 (mean reversion)
            if close[i] >= upper[i] or cci[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price touches lower BB OR CCI > 50 (mean reversion)
            if close[i] <= lower[i] or cci[i] > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals