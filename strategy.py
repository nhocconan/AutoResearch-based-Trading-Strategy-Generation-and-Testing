#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d ATR volume regime filter and 4h/1d EMA crossover trend filter.
Long when price breaks above 20-period Donchian high with 1d ATR-scaled volume > 1.5x 20-day average and EMA20 > EMA50 (4h).
Short when price breaks below 20-period Donchian low with 1d ATR-scaled volume > 1.5x 20-day average and EMA20 < EMA50 (4h).
Exit when price touches the opposite Donchian band (20-period low for long, high for short) or trend reverses.
Uses 1d for volume regime (ATR-scaled to normalize volatility), 4h for execution and trend filter.
Designed to capture volatility-expansion breakouts with institutional participation in both bull and bear markets.
Volume regime filter ensures trades only occur during periods of higher than normal volatility-adjusted participation.
Target: 25-35 trades/year per symbol to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR and volume regime calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d ATR(14) for volatility normalization
    tr1 = np.maximum(high_1d[1:] - low_1d[1:], np.abs(high_1d[1:] - close_1d[:-1]))
    tr1 = np.maximum(tr1, np.abs(low_1d[1:] - close_1d[:-1]))
    tr1 = np.concatenate([[np.nan], tr1])  # align with index 0
    atr14_1d = pd.Series(tr1).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 1d ATR-scaled volume (volume / ATR) for regime filter
    atr_scaled_vol_1d = volume_1d / atr14_1d
    atr_scaled_vol_ma_20_1d = pd.Series(atr_scaled_vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 4h Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 4h EMA20 and EMA50 for trend filter
    close_series = pd.Series(close)
    ema20_4h = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = close_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1d indicators to 4h timeframe
    atr_scaled_vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_scaled_vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 60  # need enough for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or 
            np.isnan(ema20_4h[i]) or 
            np.isnan(ema50_4h[i]) or 
            np.isnan(atr_scaled_vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume regime: current 1d ATR-scaled volume > 1.5x 20-day average (expanding volatility-adjusted participation)
        vol_regime = atr_scaled_vol_ma_20_1d_aligned[i] > 0 and volume[i] > 0 and \
                     (atr_scaled_vol_ma_20_1d_aligned[i] * 1.5) < (volume[i] / max(atr14_1d[-1] if len(atr14_1d) > 0 else 1, 1e-10))
        # Simplified: use pre-calculated ratio
        vol_ratio = (volume[i] / max(atr14_1d[-1] if len(atr14_1d) > 0 else 1, 1e-10)) if len(atr14_1d) > 0 else 0
        vol_regime = vol_ratio > (atr_scaled_vol_ma_20_1d_aligned[i] * 1.5) if not np.isnan(atr_scaled_vol_ma_20_1d_aligned[i]) else False
        
        if position == 0:
            # Long: price breaks above Donchian high with volume regime and uptrend (EMA20 > EMA50)
            if (close[i] > high_max_20[i] and 
                vol_regime and 
                ema20_4h[i] > ema50_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume regime and downtrend (EMA20 < EMA50)
            elif (close[i] < low_min_20[i] and 
                  vol_regime and 
                  ema20_4h[i] < ema50_4h[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price touches or breaks below Donchian low OR trend reverses
            if (close[i] <= low_min_20[i] or 
                ema20_4h[i] < ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price touches or breaks above Donchian high OR trend reverses
            if (close[i] >= high_max_20[i] or 
                ema20_4h[i] > ema50_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_ATRVolRegime_EMA20_50_Trend"
timeframe = "4h"
leverage = 1.0