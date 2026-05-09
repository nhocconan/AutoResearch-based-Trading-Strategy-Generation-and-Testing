#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_1wTrend_1dVolume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    """
    Elder Ray with weekly trend filter and daily volume confirmation.
    Long: Bull Power > 0, weekly close above weekly EMA(50), volume > 1.5x daily avg
    Short: Bear Power < 0, weekly close below weekly EMA(50), volume > 1.5x daily avg
    Exit: Opposite signal or weekly trend change
    Uses weekly EMA for trend, Elder Ray for momentum, daily volume for confirmation.
    Target: 15-35 trades/year on 6h timeframe.
    """
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # Get daily data for Elder Ray and volume
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Weekly EMA(50) for trend filter
    close_1w = pd.Series(df_1w['close'].values)
    ema50_1w = close_1w.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Daily EMA(13) for Elder Ray calculation
    close_1d = pd.Series(df_1d['close'].values)
    ema13_1d = close_1d.ewm(span=13, adjust=False, min_periods=13).mean().values
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Daily volume average
    vol_series = pd.Series(volume)
    vol_avg = vol_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # ensure sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(ema50_1w_aligned[i]) or np.isnan(ema13_1d_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Elder Ray components
        bull_power = high[i] - ema13_1d_aligned[i]
        bear_power = low[i] - ema13_1d_aligned[i]
        
        # Volume confirmation
        vol_ok = volume[i] > 1.5 * vol_avg[i]
        
        if position == 0:
            # Long: Bull Power positive, weekly trend up, volume confirmation
            if bull_power > 0 and close[i] > ema50_1w_aligned[i] and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: Bear Power negative, weekly trend down, volume confirmation
            elif bear_power < 0 and close[i] < ema50_1w_aligned[i] and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power negative or weekly trend down
            if bear_power < 0 or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power positive or weekly trend up
            if bull_power > 0 or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals