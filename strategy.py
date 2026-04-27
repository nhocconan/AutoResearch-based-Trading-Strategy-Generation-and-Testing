#!/usr/bin/env python3
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
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Weekly SMA(20) for trend filter
    sma20_1w = pd.Series(df_1w['close']).rolling(window=20, min_periods=20).mean().values
    sma20_1w_aligned = align_htf_to_ltf(prices, df_1w, sma20_1w)
    
    # Daily ATR(14) for volatility filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr2 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr14_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    
    # Calculate Camarilla pivot levels from previous day
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    pivot = (high_prev + low_prev + close_prev * 2) / 4
    range_ = high_prev - low_prev
    
    # Focus on R3/S3 for mean reversion entries
    r3 = pivot + range_ * 1.25
    s3 = pivot - range_ * 1.25
    
    # Align levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    atr_aligned = atr14_1d_aligned
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for all indicators
    start_idx = max(20, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(sma20_1w_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        weekly_trend = sma20_1w_aligned[i]
        atr_val = atr_aligned[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Mean reversion at R3/S3 with weekly trend filter and volatility-adjusted entry
            # Long: price touches S3, closes above it, above weekly trend, with volume spike
            # Entry only when volatility is not too high (avoid choppy markets)
            if (low[i] <= s3_aligned[i] and close[i] > s3_aligned[i] and 
                close[i] > weekly_trend and vol_spike_val and atr_val < weekly_trend * 0.05):
                signals[i] = size
                position = 1
            # Short: price touches R3, closes below it, below weekly trend, with volume spike
            elif (high[i] >= r3_aligned[i] and close[i] < r3_aligned[i] and 
                  close[i] < weekly_trend and vol_spike_val and atr_val < weekly_trend * 0.05):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: 
            # 1. Price reaches R3 (profit target)
            # 2. Price closes below weekly trend (trend change)
            # 3. Volatility spike (potential reversal)
            if (high[i] >= r3_aligned[i] or 
                close[i] < weekly_trend or 
                vol_spike_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions:
            # 1. Price reaches S3 (profit target)
            # 2. Price closes above weekly trend (trend change)
            # 3. Volatility spike (potential reversal)
            if (low[i] <= s3_aligned[i] or 
                close[i] > weekly_trend or 
                vol_spike_val):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Camarilla_R3S3_MeanReversion_WeeklyTrend_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0