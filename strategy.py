#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 4h Donchian breakout with 1d ATR regime filter and volume confirmation
    # Long: price breaks above Donchian(20) high AND ATR(14) < median ATR(50) (low volatility regime) AND volume > 1.5x avg
    # Short: price breaks below Donchian(20) low AND ATR(14) < median ATR(50) AND volume > 1.5x avg
    # Exit: price reverts to Donchian midpoint OR ATR spike (high volatility) OR volume dry-up
    # Using 4h timeframe for optimal trade frequency, Donchian for structure breakout,
    # ATR regime filter to avoid whipsaws in high volatility, volume for confirmation.
    # Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for ATR regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily ATR(14) and its 50-period median for regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range calculation
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # Align with close_1d indices
    
    # ATR(14) using Wilder's smoothing
    atr_1d = np.full_like(close_1d, np.nan)
    if len(tr) >= 15:  # Need at least 15 values for ATR(14)
        atr_1d[14] = np.nanmean(tr[1:15])  # First ATR is simple average of first 14 TR
        for i in range(15, len(tr)):
            if not np.isnan(atr_1d[i-1]) and not np.isnan(tr[i]):
                atr_1d[i] = (atr_1d[i-1] * 13 + tr[i]) / 14
    
    # Median ATR over 50 periods for regime filter
    median_atr_1d = np.full_like(close_1d, np.nan)
    for i in range(49, len(close_1d)):
        window = atr_1d[i-49:i+1]
        if not np.all(np.isnan(window)):
            median_atr_1d[i] = np.nanmedian(window)
    
    # Align daily ATR and median ATR to 4h
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    median_atr_1d_aligned = align_htf_to_ltf(prices, df_1d, median_atr_1d)
    
    # Calculate 4h Donchian channels (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_mid = np.full(n, np.nan)
    
    for i in range(lookback-1, n):
        donchian_high[i] = np.max(high[i-lookback+1:i+1])
        donchian_low[i] = np.min(low[i-lookback+1:i+1])
        donchian_mid[i] = (donchian_high[i] + donchian_low[i]) / 2
    
    # Get 4h volume for confirmation (>1.5x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(median_atr_1d_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Regime filter: ATR < median ATR = low volatility regime (favor breakouts)
        low_vol_regime = atr_1d_aligned[i] < median_atr_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_spike[i]
        
        # Entry logic: Donchian breakout + low volatility regime + volume confirmation
        long_entry = (close[i] > donchian_high[i-1]) and low_vol_regime and vol_confirm
        short_entry = (close[i] < donchian_low[i-1]) and low_vol_regime and vol_confirm
        
        # Exit logic: price reverts to midpoint OR volatility spike OR volume dry-up
        long_exit = (close[i] < donchian_mid[i]) or (atr_1d_aligned[i] > median_atr_1d_aligned[i]) or not vol_confirm
        short_exit = (close[i] > donchian_mid[i]) or (atr_1d_aligned[i] > median_atr_1d_aligned[i]) or not vol_confirm
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_1d_donchian_atr_volume_v1"
timeframe = "4h"
leverage = 1.0