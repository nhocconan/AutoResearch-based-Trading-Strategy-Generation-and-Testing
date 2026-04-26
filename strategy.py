#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_VolumeSpike_ChopRegime
Hypothesis: 4h Camarilla R3/S3 breakout with volume spike and choppy market regime filter.
- Long when price breaks above Camarilla R3 level AND volume spike AND choppy regime (mean reversion)
- Short when price breaks below Camarilla S3 level AND volume spike AND choppy regime
- Uses prior 1d range for Camarilla levels (structure-based edge)
- Volume spike confirms institutional participation (2.0x 20-period average)
- Chop regime filter (CHOP > 61.8) avoids trending markets where breakouts fail
- Designed for low frequency (target 20-50 trades/year) with proven edge on BTC/ETH
- Exit on opposite Camarilla level touch (R3 for shorts, S3 for longs)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:  # Need enough data for calculations
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for Camarilla levels and chop regime
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Camarilla levels from prior 1d bar
    # Camarilla: R3 = close + 1.1*(high-low)*1.1/4, S3 = close - 1.1*(high-low)*1.1/4
    prior_1d_high = np.roll(df_1d['high'].values, 1)
    prior_1d_low = np.roll(df_1d['low'].values, 1)
    prior_1d_close = np.roll(df_1d['close'].values, 1)
    # First value is invalid due to roll
    prior_1d_high[0] = np.nan
    prior_1d_low[0] = np.nan
    prior_1d_close[0] = np.nan
    
    cam_r3 = prior_1d_close + 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    cam_s3 = prior_1d_close - 1.1 * (prior_1d_high - prior_1d_low) * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    cam_r3_aligned = align_htf_to_ltf(prices, df_1d, cam_r3)
    cam_s3_aligned = align_htf_to_ltf(prices, df_1d, cam_s3)
    
    # Calculate Chop regime on 1d timeframe (CHOP > 61.8 = choppy/range bound)
    # Chop = 100 * log10(sum(ATR(1), n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP > 61.8 indicates ranging market (good for mean reversion)
    tr1d = np.maximum(df_1d['high'].values - df_1d['low'].values,
                      np.maximum(abs(df_1d['high'].values - np.roll(df_1d['close'].values, 1)),
                                 abs(df_1d['low'].values - np.roll(df_1d['close'].values, 1))))
    atr1d = pd.Series(tr1d).rolling(window=14, min_periods=14).mean().values
    sum_atr14 = pd.Series(atr1d).rolling(window=14, min_periods=14).sum().values
    max_high14 = pd.Series(df_1d['high'].values).rolling(window=14, min_periods=14).max().values
    min_low14 = pd.Series(df_1d['low'].values).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr14 / (np.log10(14) * (max_high14 - min_low14)))
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    choppy_regime = chop_aligned > 61.8  # Choppy/ranging market
    
    # Calculate volume spike (20-period volume average on 4h)
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma20 * 2.0)  # Volume at least 2.0x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 20 for volume MA, 1 for prior day, 14 for chop)
    start_idx = max(20, 14, 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(cam_r3_aligned[i]) or np.isnan(cam_s3_aligned[i]) or
            np.isnan(volume_spike[i]) or np.isnan(choppy_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Camarilla R3/S3 breakout conditions with volume confirmation and chop regime filter
        if position == 0:
            # Long: Price breaks above Camarilla R3 AND volume spike AND choppy regime
            if close[i] > cam_r3_aligned[i] and volume_spike[i] and choppy_regime[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Camarilla S3 AND volume spike AND choppy regime
            elif close[i] < cam_s3_aligned[i] and volume_spike[i] and choppy_regime[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: Price falls below Camarilla S3
            if close[i] < cam_s3_aligned[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: Price rises above Camarilla R3
            if close[i] > cam_r3_aligned[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0