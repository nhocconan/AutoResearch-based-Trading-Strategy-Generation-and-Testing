#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla breakout with 1d volume spike and 1w ADX trend filter
# - Uses 12h primary timeframe targeting 12-37 trades/year (50-150 total over 4 years)
# - Camarilla levels from 1d: R4/S4 for breakout continuation in trending markets
# - 1w ADX(14) > 25 ensures strong weekly trend alignment, avoiding chop
# - Volume confirmation: current 12h volume > 1.8x 24-period average for institutional participation
# - Designed to work in both bull/bear markets via weekly trend filter
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "12h_1w_camarilla_adx_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 10:
        return np.zeros(n)
    
    # Pre-compute 1w ADX(14) for trend filter
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w), 
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)), 
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(span=14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus_14 / tr_14
    di_minus = 100 * dm_minus_14 / tr_14
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Pre-compute 1d Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    rango = high_1d - low_1d
    camarilla_r3 = close_1d + (rango * 1.1 / 4)
    camarilla_s3 = close_1d - (rango * 1.1 / 4)
    camarilla_r4 = close_1d + (rango * 1.1 / 2)
    camarilla_s4 = close_1d - (rango * 1.1 / 2)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Pre-compute 12h volume confirmation
    volume_12h = prices['volume'].values
    avg_volume_24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    vol_spike = volume_12h > (1.8 * avg_volume_24)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_spike[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below S3 (mean reversion failure) or breaks below S4 (stop)
            if prices['close'].iloc[i] < camarilla_s3_aligned[i] or prices['close'].iloc[i] < camarilla_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above R3 (mean reversion failure) or breaks above R4 (stop)
            if prices['close'].iloc[i] > camarilla_r3_aligned[i] or prices['close'].iloc[i] > camarilla_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla breakout with trend and volume filters
            if vol_spike[i] and adx_aligned[i] > 25:
                # Breakout long: price closes above R4 (strong bullish breakout)
                if prices['close'].iloc[i] > camarilla_r4_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Breakout short: price closes below S4 (strong bearish breakout)
                elif prices['close'].iloc[i] < camarilla_s4_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals