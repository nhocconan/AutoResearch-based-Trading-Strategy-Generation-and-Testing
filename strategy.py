#!/usr/bin/env python3
"""
6h_ElderRay_BullBearPower_1dTrend_VolumeRegime
Hypothesis: On 6h timeframe, enter long when Elder Ray Bull Power > 0 (buying pressure) AND 1d trend is up (close > EMA50) AND volume regime is normal (not extreme). Enter short when Bear Power < 0 (selling pressure) AND 1d trend is down AND volume regime normal. Uses Elder Ray to measure bull/bear power relative to EMA13, 1d EMA50 for higher-timeframe trend filter, and volume regime filter to avoid chop/extreme volatility. Designed for low-moderate trade frequency (15-30/year) with strong edge in both bull and bear markets via trend-aligned momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    # Using 13-period EMA on 6h close
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # Volume regime: avoid extreme volatility (chop or panic)
    volume_series = pd.Series(volume)
    volume_ma = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_std = volume_series.rolling(window=20, min_periods=20).std().values
    volume_zscore = np.abs((volume - volume_ma) / np.maximum(volume_std, 1e-10))
    volume_regime_normal = volume_zscore < 2.0  # Within 2 std devs
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA13 warmup (13), volume MA/STD warmup (20)
    start_idx = max(13, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(volume_ma[i]) or np.isnan(volume_std[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Volume regime filter: only trade when volume is not extreme
        if not volume_regime_normal[i]:
            # Hold current position during extreme volume
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Elder Ray conditions
        bull_strong = bull_power[i] > 0  # Buying pressure
        bear_strong = bear_power[i] < 0  # Selling pressure
        
        # 1d trend filter
        trend_uptrend = close[i] > ema_50_1d_aligned[i]
        trend_downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: bull power positive + 1d uptrend + normal volume regime
            long_signal = bull_strong and trend_uptrend and volume_regime_normal[i]
            
            # Short: bear power negative + 1d downtrend + normal volume regime
            short_signal = bear_strong and trend_downtrend and volume_regime_normal[i]
            
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.25
            # Exit: bull power turns negative OR trend change to downtrend OR volume extreme
            if bull_power[i] <= 0 or not trend_uptrend or not volume_regime_normal[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.25
            # Exit: bear power turns positive OR trend change to uptrend OR volume extreme
            if bear_power[i] >= 0 or not trend_downtrend or not volume_regime_normal[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "6h_ElderRay_BullBearPower_1dTrend_VolumeRegime"
timeframe = "6h"
leverage = 1.0