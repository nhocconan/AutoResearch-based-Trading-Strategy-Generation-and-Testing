#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_DonchianExit_VolumeRegime
Hypothesis: Camarilla R3/S3 breakouts with volume confirmation (top 30%) exit on Donchian(20) reversal. Uses tighter R3/S3 for stronger breakouts, volume ensures participation, Donchian exit avoids giving back profits in ranging/weak markets. Fixed size 0.25 to limit trades. Target: 20-30 trades/year. Works in bull via breakouts, bears via shorting breakdowns.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for HTF filters
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Previous 12h bar's OHLC for Camarilla levels (R3/S3 = stronger breakout levels)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h_vals = df_12h['close'].values
    
    # Calculate Camarilla levels: R3, S3 (stronger breakout levels)
    rng = high_12h - low_12h
    camarilla_r3 = close_12h_vals + (rng * 1.1 / 4)   # R3 level
    camarilla_s3 = close_12h_vals - (rng * 1.1 / 4)   # S3 level
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # Volume regime: volume > 70th percentile of 50-period lookback (high volume days only)
    vol_series = pd.Series(volume)
    vol_percentile_70 = vol_series.rolling(window=50, min_periods=50).quantile(0.70).values
    volume_regime = volume > vol_percentile_70
    
    # Donchian(20) for exit: upper/lower channel
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    
    # Fixed position size to control trade frequency
    fixed_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: max of calculations (50 for EMA and volume percentile, 20 for Donchian)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(vol_percentile_70[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        camarilla_r3_val = camarilla_r3_aligned[i]
        camarilla_s3_val = camarilla_s3_aligned[i]
        vol_regime = volume_regime[i]
        donchian_up = donchian_upper[i]
        donchian_low = donchian_lower[i]
        size = fixed_size
        
        # Entry conditions: breakout of Camarilla R3/S3 with volume regime
        long_entry = (close_val > camarilla_r3_val) and vol_regime
        short_entry = (close_val < camarilla_s3_val) and vol_regime
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = size
                position = 1
            elif short_entry:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on Donchian lower break (reversal)
            if close_val < donchian_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Short - exit on Donchian upper break (reversal)
            if close_val > donchian_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_DonchianExit_VolumeRegime"
timeframe = "4h"
leverage = 1.0