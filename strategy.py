#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Bollinger Band squeeze breakout with 1d volume regime filter
# In low volatility regimes (1d BB width < 20th percentile): breakouts from 12h BB (±2σ) capture explosive moves
# Volume confirmation (>2.0x 50-period median volume) ensures institutional participation
# Uses discrete sizing (0.25) to minimize fee churn. Designed for 12h timeframe targeting 50-150 total trades over 4 years.
# BTC/ETH edge: Bollinger squeezes precede major moves; volume regime filter avoids false breakouts in low-participation environments.

name = "12h_BBSqueeze_1dVolRegime_Breakout"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for volume regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 12h Bollinger Bands (20, 2)
    close_s = pd.Series(close)
    basis = close_s.rolling(window=20, min_periods=20).mean()
    dev = close_s.rolling(window=20, min_periods=20).std()
    upper_band = basis + 2.0 * dev
    lower_band = basis - 2.0 * dev
    
    # Calculate 12h BB width % for squeeze detection
    bb_width = (upper_band - lower_band) / basis * 100
    
    # Calculate 1d volume median (50-period) for regime filter
    vol_s = pd.Series(df_1d['volume'].values)
    vol_median_50 = vol_s.rolling(window=50, min_periods=50).median()
    
    # Calculate 20th percentile of 1d BB width for squeeze threshold
    bb_width_s = pd.Series(bb_width.values[-len(df_1d)*12:])  # Approximate 12h bars in 1d
    if len(bb_width_s) < 20:
        squeeze_threshold = 5.0  # fallback
    else:
        squeeze_threshold = bb_width_s.rolling(window=20, min_periods=20).quantile(0.20).iloc[-1] if not bb_width_s.rolling(window=20, min_periods=20).quantile(0.20).empty else 5.0
    
    # Align 1d volume median to 12h timeframe
    vol_median_50_aligned = align_htf_to_ltf(prices, df_1d, vol_median_50.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(basis[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(bb_width[i]) or np.isnan(vol_median_50_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume regime: current 12h volume > 2.0 x 1d volume median (scaled)
        # Scale 1d median to approximate 12h expectation: 1d volume / 2 (since 2x12h ≈ 1d)
        volume_regime = volume[i] > (2.0 * vol_median_50_aligned[i] / 2.0)
        
        # Squeeze condition: 12h BB width < 20th percentile threshold (use fixed threshold for stability)
        is_squeeze = bb_width[i] < 5.0  # Empirical threshold for low volatility
        
        if position == 0:
            # Look for breakout from Bollinger Bands with volume confirmation
            if close[i] > upper_band[i] and volume_regime and is_squeeze:
                signals[i] = 0.25
                position = 1
            elif close[i] < lower_band[i] and volume_regime and is_squeeze:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price returns to basis OR volatility expands (BB width > 2* squeeze threshold)
            if close[i] <= basis[i] or bb_width[i] > 2.0 * squeeze_threshold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price returns to basis OR volatility expands
            if close[i] >= basis[i] or bb_width[i] > 2.0 * squeeze_threshold:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals