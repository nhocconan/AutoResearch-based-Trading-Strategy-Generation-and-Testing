#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
# Williams %R(14) from 1d: > -20 = overbought (short bias), < -80 = oversold (long bias)
# Trade logic: Long when Bull Power > 0 AND Williams %R < -80 (oversold in bearish regime)
# Short when Bear Power < 0 AND Williams %R > -20 (overbought in bullish regime)
# Uses discrete sizing 0.25 to minimize fee churn
# Works in bull markets (buy oversold dips) and bear markets (sell overbought rallies)
# Volume confirmation (>1.5x 20 EMA volume) ensures institutional participation
# Target: 80-180 total trades over 4 years = 20-45/year for 6h timeframe

name = "6h_ElderRay_WilliamsR_VolumeSpike"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Williams %R regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need enough data for Williams %R calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) from 1d: (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close_1d) / (highest_high - lowest_low)
    # Shift by 1 to use only prior completed 1d bar (no look-ahead)
    williams_r_shifted = np.roll(williams_r, 1)
    williams_r_shifted[0] = np.nan
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r_shifted)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Elder Ray components
    bull_power = high - ema_13  # High - EMA13
    bear_power = low - ema_13   # Low - EMA13
    
    # Volume confirmation: 20-period EMA of volume on 6h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 (bullish momentum) AND Williams %R < -80 (oversold) AND volume spike
            if bull_power[i] > 0 and williams_r_aligned[i] < -80 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 (bearish momentum) AND Williams %R > -20 (overbought) AND volume spike
            elif bear_power[i] < 0 and williams_r_aligned[i] > -20 and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 OR Williams %R >= -50 (recovering from oversold)
            if bull_power[i] <= 0 or williams_r_aligned[i] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 OR Williams %R <= -50 (declining from overbought)
            if bear_power[i] >= 0 or williams_r_aligned[i] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals