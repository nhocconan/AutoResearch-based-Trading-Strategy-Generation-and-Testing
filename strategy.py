#!/usr/bin/env python3
"""
1h_HMA_Trend_VolumeSpike
Hypothesis: On 1h timeframe, use 4h HMA(21) for trend direction and 1d volume spike for entry timing.
Only trade during 08-20 UTC session to avoid low-liquidity hours.
HMA reduces lag while maintaining smoothness. Volume spike confirms institutional interest.
Targets 15-37 trades/year (60-150 total over 4 years) by requiring confluence of 4h trend, 1d volume, and session.
Works in bull/bear markets via 4h trend filter (avoid counter-trend trades).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Precompute session filter (08-20 UTC) ONCE before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Load 4h data ONCE before loop for HMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h HMA(21) for trend
    close_4h = df_4h['close'].values
    # HMA = WMA(2*WMA(n/2) - WMA(n)), sqrt(n))
    def wma(values, window):
        weights = np.arange(1, window + 1)
        return np.convolve(values, weights, 'valid') / weights.sum()
    
    n_hl = 21
    half_n = n_hl // 2
    sqrt_n = int(np.sqrt(n_hl))
    
    if len(close_4h) < n_hl:
        hma_4h = np.full_like(close_4h, np.nan)
    else:
        wma_half = wma(close_4h, half_n)
        wma_full = wma(close_4h, n_hl)
        # Pad to match original length
        wma_half_padded = np.concatenate([np.full(half_n, np.nan), wma_half])
        wma_full_padded = np.concatenate([np.full(n_hl-1, np.nan), wma_full])
        raw_hma = 2 * wma_half_padded - wma_full_padded
        hma_4h = wma(raw_hma, sqrt_n)
        # Final padding for HMA
        hma_4h = np.concatenate([np.full(sqrt_n-1, np.nan), hma_4h[:len(close_4h)-sqrt_n+1]])
    
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # Load 1d data ONCE before loop for volume spike
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d volume median(20) for spike detection
    vol_1d = df_1d['volume'].values
    vol_median_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).median().values
    vol_median_20_aligned = align_htf_to_ltf(prices, df_1d, vol_median_20)
    
    # Reduced fixed position size to control trade frequency and drawdown
    fixed_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need 30 for 4h HMA, 20 for 1d volume median
    start_idx = max(30, 20)
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
            
        # Skip if any data not ready
        if (np.isnan(hma_4h_aligned[i]) or
            np.isnan(vol_median_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        hma_4h_val = hma_4h_aligned[i]
        vol_median = vol_median_20_aligned[i]
        vol_spike = volume[i] > (1.5 * vol_median)
        
        if position == 0:
            # Flat - look for entry
            # Only long in uptrend (close > HMA), only short in downtrend (close < HMA)
            long_entry = (close_val > hma_4h_val) and vol_spike
            short_entry = (close_val < hma_4h_val) and vol_spike
            
            if long_entry:
                signals[i] = fixed_size
                position = 1
                entry_price = close_val
            elif short_entry:
                signals[i] = -fixed_size
                position = -1
                entry_price = close_val
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on trend reversal (close < HMA)
            if close_val < hma_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = fixed_size
        elif position == -1:
            # Short - exit on trend reversal (close > HMA)
            if close_val > hma_4h_val:
                signals[i] = 0.0
                position = 0
                entry_price = 0.0
            else:
                signals[i] = -fixed_size
    
    return signals

name = "1h_HMA_Trend_VolumeSpike"
timeframe = "1h"
leverage = 1.0