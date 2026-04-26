#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeRegime
Hypothesis: Camarilla R3/S3 breakout on 4h with 1d EMA50 trend filter, volume confirmation (>1.5x 20-period MA), and chop regime filter (CHOP < 61.8 = trending). 
Long when price breaks above R3 in uptrend with volume spike and trending regime. 
Short when price breaks below S3 in downtrend with volume spike and trending regime.
Uses discrete position sizing (0.25) to minimize fee churn. 
Camarilla levels derived from prior 1d OHLC. 
Designed to work in both bull and bear markets by following the 1d trend and avoiding range-bound markets.
Target: 19-50 trades/year (75-200 total over 4 years) by using stricter R3/S3 levels and regime filter.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d OHLC
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Shift by 1 to use prior day's OHLC for current day's levels
    close_1d_prev = np.roll(close_1d, 1)
    high_1d_prev = np.roll(high_1d, 1)
    low_1d_prev = np.roll(low_1d, 1)
    close_1d_prev[0] = np.nan
    high_1d_prev[0] = np.nan
    low_1d_prev[0] = np.nan
    
    # Camarilla R3, S3 levels (stronger breakout levels)
    # R3 = Close + (High - Low) * 1.1/4
    # S3 = Close - (High - Low) * 1.1/4
    camarilla_range = high_1d_prev - low_1d_prev
    r3 = close_1d_prev + camarilla_range * 1.1 / 4
    s3 = close_1d_prev - camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1d EMA50 trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    uptrend_1d = close > ema_50_1d_aligned
    downtrend_1d = close < ema_50_1d_aligned
    
    # Volume confirmation: volume > 1.5x 20-period MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    # Choppiness regime filter: CHOP < 61.8 = trending (favor trend following)
    # CHOP = 100 * log10(sum(ATR(14),14) / (max(high,14) - min(low,14))) / log10(14)
    atr_14 = pd.Series(np.maximum(high - low, np.maximum(high - np.roll(close, 1), np.roll(close, 1) - low))).rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    trending_regime = chop < 61.8  # Only trade in trending markets
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need 50 for 1d EMA + 20 for volume MA + 14 for ATR + 1 for Camarilla shift)
    start_idx = 65
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_spike[i]) or np.isnan(trending_regime[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        if position == 0:
            # Long: price breaks above R3 with 1d uptrend, volume spike, and trending regime
            if (close[i] > r3_aligned[i] and 
                uptrend_1d[i] and volume_spike[i] and trending_regime[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3 with 1d downtrend, volume spike, and trending regime
            elif (close[i] < s3_aligned[i] and 
                  downtrend_1d[i] and volume_spike[i] and trending_regime[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: price closes below S3 (strong reversal) OR 1d trend changes to downtrend OR regime changes to ranging
            if (close[i] < s3_aligned[i] or not uptrend_1d[i] or not trending_regime[i]):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: price closes above R3 (strong reversal) OR 1d trend changes to uptrend OR regime changes to ranging
            if (close[i] > r3_aligned[i] or not downtrend_1d[i] or not trending_regime[i]):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_VolumeRegime"
timeframe = "4h"
leverage = 1.0