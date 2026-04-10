#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot levels from 1d + volume spike + ADX regime filter
# - Long: price breaks above R4 (1.5x ATR above close) with volume > 2x average AND ADX(14) > 25 (trending)
# - Short: price breaks below S4 (1.5x ATR below close) with volume > 2x average AND ADX(14) > 25 (trending)
# - Exit: price reverts to R3/S3 levels OR ADX drops below 20 (range regime)
# - Uses discrete position sizing (0.25) to minimize fee churn
# - Designed for 6h timeframe: targets 12-37 trades/year to avoid fee drag
# - Works in bull/bear markets: ADX filter ensures we only trade strong trends, Camarilla breakouts capture momentum

name = "6h_1d_camarilla_breakout_adx_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 1d Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    # Camarilla levels: R4 = close + range * 1.5, S4 = close - range * 1.5
    camarilla_r4 = close_1d + (range_1d * 1.5)
    camarilla_s4 = close_1d - (range_1d * 1.5)
    camarilla_r3 = close_1d + (range_1d * 1.25)
    camarilla_s3 = close_1d - (range_1d * 1.25)
    
    # Align Camarilla levels to 6h
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Pre-compute 6h ADX(14) for regime filter
    high_6h = prices['high'].values
    low_6h = prices['low'].values
    close_6h = prices['close'].values
    
    # True Range
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_6h - np.roll(high_6h, 1)) > (np.roll(low_6h, 1) - low_6h), 
                       np.maximum(high_6h - np.roll(high_6h, 1), 0), 0)
    dm_minus = np.where((np.roll(low_6h, 1) - low_6h) > (high_6h - np.roll(high_6h, 1)), 
                        np.maximum(np.roll(low_6h, 1) - low_6h, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # DI and DX
    di_plus = np.where(tr_14 != 0, (dm_plus_14 / tr_14) * 100, 0)
    di_minus = np.where(tr_14 != 0, (dm_minus_14 / tr_14) * 100, 0)
    dx = np.where((di_plus + di_minus) != 0, 
                  np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100, 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Pre-compute 6h volume confirmation
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume_6h > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(camarilla_r4_aligned[i]) or np.isnan(camarilla_s4_aligned[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(adx[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reverts to R3 OR ADX drops below 20 (range regime)
            if close_6h[i] <= camarilla_r3_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reverts to S3 OR ADX drops below 20 (range regime)
            if close_6h[i] >= camarilla_s3_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with volume and ADX filters
            if vol_spike[i] and adx[i] > 25:
                # Long: price breaks above R4 with volume spike AND strong trend
                if close_6h[i] > camarilla_r4_aligned[i]:
                    position = 1
                    entry_price = close_6h[i]
                    signals[i] = 0.25
                # Short: price breaks below S4 with volume spike AND strong trend
                elif close_6h[i] < camarilla_s4_aligned[i]:
                    position = -1
                    entry_price = close_6h[i]
                    signals[i] = -0.25
    
    return signals