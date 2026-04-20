#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data once for pivot levels and volume
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 20:
        return np.zeros(n)
    
    high_daily = df_daily['high'].values
    low_daily = df_daily['low'].values
    close_daily = df_daily['close'].values
    volume_daily = df_daily['volume'].values
    
    # Calculate classic pivot point (PP) and support/resistance levels
    # PP = (H + L + C) / 3
    # R1 = 2*PP - L
    # S1 = 2*PP - H
    # R2 = PP + (H - L)
    # S2 = PP - (H - L)
    # R3 = H + 2*(PP - L)
    # S3 = L - 2*(H - PP)
    pp_daily = (high_daily + low_daily + close_daily) / 3.0
    r1_daily = 2 * pp_daily - low_daily
    s1_daily = 2 * pp_daily - high_daily
    r2_daily = pp_daily + (high_daily - low_daily)
    s2_daily = pp_daily - (high_daily - low_daily)
    r3_daily = high_daily + 2 * (pp_daily - low_daily)
    s3_daily = low_daily - 2 * (high_daily - pp_daily)
    
    # Align pivot levels to 6h timeframe (use previous day's levels)
    pp_daily_aligned = align_htf_to_ltf(prices, df_daily, pp_daily)
    r1_daily_aligned = align_htf_to_ltf(prices, df_daily, r1_daily)
    s1_daily_aligned = align_htf_to_ltf(prices, df_daily, s1_daily)
    r2_daily_aligned = align_htf_to_ltf(prices, df_daily, r2_daily)
    s2_daily_aligned = align_htf_to_ltf(prices, df_daily, s2_daily)
    r3_daily_aligned = align_htf_to_ltf(prices, df_daily, r3_daily)
    s3_daily_aligned = align_htf_to_ltf(prices, df_daily, s3_daily)
    
    # Daily volume average (20) for confirmation
    vol_ma_daily = pd.Series(volume_daily).rolling(window=20, min_periods=20).mean().values
    vol_ma_daily_aligned = align_htf_to_ltf(prices, df_daily, vol_ma_daily)
    
    # Main timeframe data (6h)
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if NaN in critical values
        if (np.isnan(pp_daily_aligned[i]) or np.isnan(r1_daily_aligned[i]) or 
            np.isnan(s1_daily_aligned[i]) or np.isnan(vol_ma_daily_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        pp = pp_daily_aligned[i]
        r1 = r1_daily_aligned[i]
        s1 = s1_daily_aligned[i]
        r2 = r2_daily_aligned[i]
        s2 = s2_daily_aligned[i]
        r3 = r3_daily_aligned[i]
        s3 = s3_daily_aligned[i]
        vol_ma = vol_ma_daily_aligned[i]
        vol_current = volume[i]
        
        # Volume filter: current volume > 1.3x daily average
        vol_ok = vol_current > 1.3 * vol_ma
        
        if position == 0:
            # Long breakout: price breaks above R1 with volume
            if price > r1 and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short breakdown: price breaks below S1 with volume
            elif price < s1 and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls back below PP (mean reversion) or breaks S2 (stop)
            if price < pp or price < s2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises back above PP (mean reversion) or breaks R2 (stop)
            if price > pp or price > r2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1d_Pivot_R1S1_Breakout_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0