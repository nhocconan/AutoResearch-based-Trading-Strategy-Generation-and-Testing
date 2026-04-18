#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d volume confirmation and ADX trend filter.
# Long when price breaks above Donchian(20) high + 1d volume > 1.5x 20-period average + ADX(14) > 25.
# Short when price breaks below Donchian(20) low + 1d volume > 1.5x 20-period average + ADX(14) > 25.
# Exit when price crosses back through Donchian(20) midpoint.
# Uses Donchian channel for breakout, volume for conviction, ADX for trend strength.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_1dVolume_ADX_Filter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for volume and ADX
    df_1d = get_htf_data(prices, '1d')
    
    # 1d volume MA(20)
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    # 1d ADX(14)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - close_1d)
    tr3 = np.abs(low_1d - close_1d)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    dm_plus = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    dm_minus = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    atr_1d_safe = np.where(atr_1d == 0, 1e-10, atr_1d)
    di_plus = 100 * pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values / atr_1d_safe
    di_minus = 100 * pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values / atr_1d_safe
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
    adx_1d = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) on 4h
    dc_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    dc_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    # Volume filter: current 1d volume > 1.5 * 20-period average
    volume_filter = vol_1d > (1.5 * vol_ma_20)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1d, volume_filter)
    
    # ADX filter: ADX > 25
    adx_filter = adx_1d > 25
    adx_filter_aligned = align_htf_to_ltf(prices, df_1d, adx_filter)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(adx_1d_aligned[i]) or
            np.isnan(volume_filter_aligned[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        dc_high_val = dc_high[i]
        dc_low_val = dc_low[i]
        dc_mid_val = dc_mid[i]
        vol_filter = volume_filter_aligned[i]
        adx_filter = adx_filter_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume and trend
            if close_val > dc_high_val and vol_filter and adx_filter:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and trend
            elif close_val < dc_low_val and vol_filter and adx_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below midpoint
            if close_val < dc_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above midpoint
            if close_val > dc_mid_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals