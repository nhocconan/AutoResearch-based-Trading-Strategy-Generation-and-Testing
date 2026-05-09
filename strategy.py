#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Choppiness Index (CI) regime filter + 1w EMA trend + volume confirmation
# Uses CI to detect ranging (CI > 61.8) vs trending (CI < 38.2) markets on 1d.
# In ranging markets: mean reversion at Bollinger Bands (20, 2.0) with volume confirmation.
# In trending markets: follow 1w EMA50 direction with breakout confirmation.
# Designed for 1d timeframe with target of 30-100 trades over 4 years (7-25/year).
# Works in bull/bear markets by adapting to regime.
name = "1d_ChoppinessIndex_Regime_EMA50_Volume"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Choppiness Index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d ATR(14) for Choppiness Index
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d true range sum and price range for Choppiness Index
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    high_max = df_1d['high'].rolling(window=14, min_periods=14).max().values
    low_min = df_1d['low'].rolling(window=14, min_periods=14).min().values
    price_range = high_max - low_min
    
    # Calculate Choppiness Index: CI = 100 * log10(atr_sum / price_range) / log10(14)
    # Avoid division by zero and log of zero
    ci_raw = 100 * np.log10(atr_sum / (price_range + 1e-10)) / np.log10(14)
    ci_raw = np.where((price_range > 0) & (atr_sum > 0), ci_raw, 50.0)  # default to neutral
    ci_1d = ci_raw  # already aligned to 1d
    
    # Align Choppiness Index to 1d (same timeframe, no alignment needed)
    ci = ci_1d
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 trend filter
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Bollinger Bands (20, 2.0) on 1d
    sma_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    std_20 = pd.Series(close).rolling(window=20, min_periods=20).std().values
    upper_bb = sma_20 + 2.0 * std_20
    lower_bb = sma_20 - 2.0 * std_20
    
    # Volume filter: current volume > 1.3x 20-period average volume
    avg_volume = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * avg_volume)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for calculations
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ci[i]) or np.isnan(ema_50_1d[i]) or np.isnan(sma_20[i]) or 
            np.isnan(std_20[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Regime detection
        is_ranging = ci[i] > 61.8  # Choppy/ranging market
        is_trending = ci[i] < 38.2  # Trending market
        
        if position == 0:
            # Enter long in ranging market at lower BB with volume
            if is_ranging and close[i] <= lower_bb[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Enter short in ranging market at upper BB with volume
            elif is_ranging and close[i] >= upper_bb[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            # Enter long in trending market when price > EMA50 and rising
            elif is_trending and close[i] > ema_50_1d[i] and close[i] > close[i-1]:
                signals[i] = 0.25
                position = 1
            # Enter short in trending market when price < EMA50 and falling
            elif is_trending and close[i] < ema_50_1d[i] and close[i] < close[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: opposite BB touch or trend reversal
            if is_ranging and close[i] >= upper_bb[i]:
                signals[i] = 0.0
                position = 0
            elif is_trending and close[i] < ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: opposite BB touch or trend reversal
            if is_ranging and close[i] <= lower_bb[i]:
                signals[i] = 0.0
                position = 0
            elif is_trending and close[i] > ema_50_1d[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals