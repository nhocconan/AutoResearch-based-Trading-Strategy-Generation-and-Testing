#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot mean reversion with 1w ADX trend filter and volume spike filter
# - Camarilla levels from 1d: fade at R3/S3 (mean reversion) in ranging markets
# - 1w ADX(14) < 20 to identify ranging/weak trend conditions for mean reversion to work
# - Volume confirmation: current 6h volume < 0.5x 20-period average to avoid low-liquidity false signals
# - Designed for 6h timeframe: targets 12-37 trades/year (50-150 total over 4 years) to avoid fee drag
# - Works in bull/bear markets: weekly ADX filter ensures we only mean revert in choppy/range-bound markets
# - Uses discrete position sizing (0.25) to minimize fee churn

name = "6h_1w_camarilla_meanrev_adxvol_v1"
timeframe = "6h"
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
    
    # Pre-compute 1w ADX(14) for trend filter (ranging market when ADX < 20)
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
    
    # Pre-compute 6h volume confirmation (low volume for mean reversion)
    volume_6h = prices['volume'].values
    avg_volume_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    vol_low = volume_6h < (0.5 * avg_volume_20)  # Low volume for cleaner mean reversion
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (mean reversion target) or breaks above R4 (trend change)
            if prices['close'].iloc[i] >= camarilla_r3_aligned[i] or prices['close'].iloc[i] > camarilla_r4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 (mean reversion target) or breaks below S4 (trend change)
            if prices['close'].iloc[i] <= camarilla_s3_aligned[i] or prices['close'].iloc[i] < camarilla_s4_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Camarilla mean reversion opportunities in ranging markets
            if vol_low[i] and adx_aligned[i] < 20:  # Low volume + ranging market
                # Mean reversion long: price touches or crosses S3 (support)
                if prices['close'].iloc[i] <= camarilla_s3_aligned[i]:
                    position = 1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = 0.25
                # Mean reversion short: price touches or crosses R3 (resistance)
                elif prices['close'].iloc[i] >= camarilla_r3_aligned[i]:
                    position = -1
                    entry_price = prices['close'].iloc[i]
                    signals[i] = -0.25
    
    return signals