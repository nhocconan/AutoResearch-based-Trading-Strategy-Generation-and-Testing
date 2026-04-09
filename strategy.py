#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot levels from 1d + volume confirmation + chop regime filter
# - Primary signal: Price touches Camarilla H3 (resistance) for short or L3 (support) for long on 4h
# - Trend filter: 12h ADX < 25 (range market) - Camarilla works best in ranging conditions
# - Volume confirmation: 4h volume > 1.5x 20-period median volume (avoid false breakouts)
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 20-50 trades/year (75-200 total over 4 years) per 4h strategy guidelines
# - Works in bull/bear: Camarilla pivots effective in ranges, ADX filter avoids trending markets where pivots fail

name = "4h_12h_camarilla_adx_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # Pre-compute 12h ADX for regime filter (range market: ADX < 25)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # True Range
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr_12h = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_12h[1:] - high_12h[:-1]) > (low_12h[:-1] - low_12h[1:]),
                       np.maximum(high_12h[1:] - high_12h[:-1], 0), 0)
    dm_minus = np.where((low_12h[:-1] - low_12h[1:]) > (high_12h[1:] - high_12h[:-1]),
                        np.maximum(low_12h[:-1] - low_12h[1:], 0), 0)
    dm_plus = np.concatenate([[0], dm_plus])
    dm_minus = np.concatenate([[0], dm_minus])
    
    # Smoothed DM
    dm_plus_smooth = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # DI+ and DI-
    di_plus = 100 * dm_plus_smooth / (atr_12h + 1e-10)
    di_minus = 100 * dm_minus_smooth / (atr_12h + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx_12h = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_12h_aligned = align_htf_to_ltf(prices, df_12h, adx_12h)
    
    # Pre-compute 1d OHLC for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low), etc.
    # L4 = close - 1.5*(high-low), L3 = close - 1.1*(high-low), etc.
    range_1d = high_1d - low_1d
    camarilla_h3 = close_1d + 1.1 * range_1d  # Resistance
    camarilla_l3 = close_1d - 1.1 * range_1d  # Support
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # 4h volume regime: volume > 1.5x 20-period median volume
    volume = prices['volume'].values
    median_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    volume_regime = volume > (1.5 * median_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_12h_aligned[i]) or
            np.isnan(camarilla_h3_aligned[i]) or
            np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price crosses above Camarilla H3 (resistance) OR ADX > 25 (trending market)
            if (prices['close'].iloc[i] >= camarilla_h3_aligned[i] or 
                adx_12h_aligned[i] > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses below Camarilla L3 (support) OR ADX > 25 (trending market)
            if (prices['close'].iloc[i] <= camarilla_l3_aligned[i] or 
                adx_12h_aligned[i] > 25):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla level touches with volume confirmation and range filter
            # Long: price touches or crosses above Camarilla L3 (support) AND volume regime AND ADX < 25 (range)
            if (prices['close'].iloc[i] >= camarilla_l3_aligned[i] and 
                volume_regime[i] and 
                adx_12h_aligned[i] < 25):
                position = 1
                signals[i] = 0.25
            # Short: price touches or crosses below Camarilla H3 (resistance) AND volume regime AND ADX < 25 (range)
            elif (prices['close'].iloc[i] <= camarilla_h3_aligned[i] and 
                  volume_regime[i] and 
                  adx_12h_aligned[i] < 25):
                position = -1
                signals[i] = -0.25
    
    return signals