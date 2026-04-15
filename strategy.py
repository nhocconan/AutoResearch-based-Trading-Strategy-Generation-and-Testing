#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with volume confirmation and ADX trend filter
# Uses daily Camarilla pivot levels (resistance/support) for mean reversion entries,
# volume to confirm rejection at levels, and ADX to avoid strong trends where reversals fail.
# Works in both bull and bear by fading extremes in ranging markets while respecting trend.
# Target: 60-120 total trades over 4 years (15-30/year) with high-conviction entries.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 4h data (primary timeframe) for price action and trend
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Load 1d data for Camarilla pivots and ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels from previous day
    # R4 = Close + 1.5 * (High - Low)
    # R3 = Close + 1.1 * (High - Low)
    # R2 = Close + 0.6 * (High - Low)
    # R1 = Close + 0.38 * (High - Low)
    # PP = (High + Low + Close) / 3
    # S1 = Close - 0.38 * (High - Low)
    # S2 = Close - 0.6 * (High - Low)
    # S3 = Close - 1.1 * (High - Low)
    # S4 = Close - 1.5 * (High - Low)
    cam_r4 = close_1d + 1.5 * (high_1d - low_1d)
    cam_r3 = close_1d + 1.1 * (high_1d - low_1d)
    cam_r2 = close_1d + 0.6 * (high_1d - low_1d)
    cam_r1 = close_1d + 0.38 * (high_1d - low_1d)
    cam_pp = (high_1d + low_1d + close_1d) / 3.0
    cam_s1 = close_1d - 0.38 * (high_1d - low_1d)
    cam_s2 = close_1d - 0.6 * (high_1d - low_1d)
    cam_s3 = close_1d - 1.1 * (high_1d - low_1d)
    cam_s4 = close_1d - 1.5 * (high_1d - low_1d)
    
    # Calculate ADX (14-period) on 1d for trend strength
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smooth TR, DM+, DM-
    tr14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus14 = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus14 = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = 100 * dm_plus14 / (tr14 + 1e-10)
    di_minus = 100 * dm_minus14 / (tr14 + 1e-10)
    
    # DX and ADX
    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period on 1d)
    vol_avg_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 4h timeframe
    cam_r4_aligned = align_htf_to_ltf(prices, df_1d, cam_r4)
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_r2_aligned = align_htf_to_ltf(prices, df_1d, cam_r2)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_pp_aligned = align_htf_to_ltf(prices, df_1d, cam_pp)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    cam_s2_aligned = align_htf_to_ltf(prices, df_1d, cam_s2)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    cam_s4_aligned = align_htf_to_ltf(prices, df_1d, cam_s4)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    vol_avg_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(cam_r1_aligned[i]) or np.isnan(cam_s1_aligned[i]) or
            np.isnan(adx_aligned[i]) or np.isnan(vol_avg_aligned[i])):
            continue
        
        # Long entry: price rejects S1/S2 with volume + weak trend (ADX < 25)
        if ((close[i] <= cam_s1_aligned[i] * 1.002 or close[i] <= cam_s2_aligned[i] * 1.002) and
            volume[i] > 1.5 * vol_avg_aligned[i] and
            adx_aligned[i] < 25 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: price rejects R1/R2 with volume + weak trend (ADX < 25)
        elif ((close[i] >= cam_r1_aligned[i] * 0.998 or close[i] >= cam_r2_aligned[i] * 0.998) and
              volume[i] > 1.5 * vol_avg_aligned[i] and
              adx_aligned[i] < 25 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: price reaches pivot point or ADX strengthens (trend emerging)
        elif position == 1 and (close[i] >= cam_pp_aligned[i] * 0.998 or adx_aligned[i] > 30):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= cam_pp_aligned[i] * 1.002 or adx_aligned[i] > 30):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_Pivot_Volume_ADX_Filter"
timeframe = "4h"
leverage = 1.0