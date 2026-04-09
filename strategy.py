#!/usr/bin/env python3
# 12h_hma_trend_volume_v1
# Hypothesis: 12h strategy using Hull Moving Average (HMA) trend with volume confirmation.
# HMA reduces lag while maintaining smoothness for trend identification. Volume confirms
# institutional participation in moves. Works in both bull/bear by following the trend.
# Discrete sizing (0.0, ±0.25) minimizes fee churn. Target: 12-37 trades/year.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_hma_trend_volume_v1"
timeframe = "12h"
leverage = 1.0

def _hma(arr, period):
    """Calculate Hull Moving Average."""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=np.float64)
    half = period // 2
    sqrt = int(np.sqrt(period))
    arr = np.asarray(arr, dtype=np.float64)
    wma2 = pd.Series(arr).ewm(span=half, adjust=False).mean()
    wma1 = pd.Series(arr).ewm(span=period, adjust=False).mean()
    raw = 2 * wma2 - wma1
    hma = pd.Series(raw).ewm(span=sqrt, adjust=False).mean()
    return hma.values

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d HTF data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate HMA (55-period) on close
    hma_55 = _hma(close, 55)
    
    # Calculate 1d average volume for regime filter
    vol_1d = df_1d['volume'].values
    vol_ma_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(hma_55[i]) or np.isnan(hma_55[i-1]) or 
            np.isnan(vol_ma_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 12h volume > 1.2x 1d average volume (scaled)
        # Scale 1d average to 12h by dividing by 2 (approx 2x 12h bars in 1d)
        volume_regime = volume[i] > 1.2 * (vol_ma_1d_aligned[i] / 2.0)
        
        if position == 1:  # Long position
            # Exit: price closes below HMA
            if close[i] < hma_55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above HMA
            if close[i] > hma_55[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            if volume_regime:
                # Long entry: price closes above HMA and HMA is rising
                if close[i] > hma_55[i] and hma_55[i] > hma_55[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Short entry: price closes below HMA and HMA is falling
                elif close[i] < hma_55[i] and hma_55[i] < hma_55[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals