#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Williams %R + 1d ADX trend filter with volume spike confirmation
# - Williams %R(14) on 6h for mean reversion entries: long when < -80, short when > -20
# - 1d ADX(14) > 25 confirms strong trend direction (avoid counter-trend in weak markets)
# - Volume spike: 6h volume > 2.0x 20-period average to confirm momentum
# - Only take longs in uptrend (ADX>25) when Williams %R < -80 (oversold in uptrend)
# - Only take shorts in downtrend (ADX>25) when Williams %R > -20 (overbought in downtrend)
# - Exit when Williams %R reverts to > -50 (long) or < -50 (short) or volume drops
# - Position size: 0.25 (25% of capital) - discrete level to minimize fee churn
# - Target: ~12-37 trades/year (50-150 total over 4 years) per 6h strategy guidelines
# - Works in both bull/bear: ADX filter ensures we only trade with strong trends,
#   Williams %R provides precise mean-reversion entries within the trend

name = "6h_1d_williams_adx_volume_v1"
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
    
    # Pre-compute 1d indicators
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX(14)
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr_1d[0]
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Movement
    dm_plus = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d),
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)),
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed DM and TR
    dm_plus_smooth = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_smooth = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    atr_smooth = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Directional Indicators
    di_plus = np.where(atr_smooth > 0, 100 * dm_plus_smooth / atr_smooth, 0)
    di_minus = np.where(atr_smooth > 0, 100 * dm_minus_smooth / atr_smooth, 0)
    
    # DX and ADX
    dx = np.where((di_plus + di_minus) > 0, 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus), 0)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Align ADX to 6h timeframe (completed 1d bar only)
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 6h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R(14) on 6h
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = np.where((highest_high_14 - lowest_low_14) > 0,
                          -100 * (highest_high_14 - close) / (highest_high_14 - lowest_low_14), -50)
    
    # 6h volume > 2.0x 20-period average (volume confirmation)
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * avg_volume_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(williams_r[i]) or
            np.isnan(volume_spike[i]) or
            adx_aligned[i] < 25):  # Require strong trend
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R reverts above -50 OR volume drops
            if williams_r[i] > -50 or not volume_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Williams %R reverts below -50 OR volume drops
            if williams_r[i] < -50 or not volume_spike[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Williams %R extreme with volume spike in direction of trend
            # Long: Williams %R < -80 (oversold) AND volume spike in uptrend (ADX>25)
            if williams_r[i] < -80 and volume_spike[i]:
                position = 1
                signals[i] = 0.25
            # Short: Williams %R > -20 (overbought) AND volume spike in downtrend (ADX>25)
            elif williams_r[i] > -20 and volume_spike[i]:
                position = -1
                signals[i] = -0.25
    
    return signals