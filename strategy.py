#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout + 1w EMA200 trend filter + volume confirmation
# - Primary signal: 12h price breaks above/below 20-period Donchian channel
# - Trend filter: 1w EMA200 - price must be above EMA for longs, below for shorts
# - Volume confirmation: 12h volume > 30-period median volume (avoid low-participation signals)
# - Exit: price retracement to midpoint of Donchian channel OR opposite breakout
# - Position size: 0.25 (discrete level) to minimize fee churn
# - Target: 12-37 trades/year (50-150 total over 4 years) per 12h strategy guidelines
# - Works in bull/bear: Donchian captures breakouts, EMA200 filter ensures alignment with
#   higher timeframe trend, reducing false signals in strong counter-trend moves

name = "12h_1w_donchian_ema_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 210:
        return np.zeros(n)
    
    # Pre-compute 1w indicators
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # 1w EMA200 for trend direction
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 12h timeframe (completed 1w bar only)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # 12h price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (highest_high_20 + lowest_low_20) / 2.0
    
    # 12h volume regime: volume > 30-period median volume
    median_volume_30 = pd.Series(volume).rolling(window=30, min_periods=30).median().values
    volume_regime = volume > median_volume_30
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_200_aligned[i]) or
            np.isnan(highest_high_20[i]) or
            np.isnan(lowest_low_20[i]) or
            np.isnan(volume_regime[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price retracement to Donchian midpoint OR opposite breakout below lower band
            if close[i] <= donchian_mid[i] or close[i] < lowest_low_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price retracement to Donchian midpoint OR opposite breakout above upper band
            if close[i] >= donchian_mid[i] or close[i] > highest_high_20[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Look for Donchian breakouts with volume confirmation and EMA200 filter
            # Long: price breaks above upper Donchian band AND volume regime AND price above EMA200
            if close[i] > highest_high_20[i] and volume_regime[i] and close[i] > ema_200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below lower Donchian band AND volume regime AND price below EMA200
            elif close[i] < lowest_low_20[i] and volume_regime[i] and close[i] < ema_200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals