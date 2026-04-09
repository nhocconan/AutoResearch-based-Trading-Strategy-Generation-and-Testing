#!/usr/bin/env python3
# 6h_elder_ray_regime_volume_v1
# Hypothesis: 6h strategy using Elder Ray Index (Bull/Bear Power) with EMA13 trend filter and volume confirmation.
# Long: Bull Power > 0, Bear Power < 0, close > EMA13, volume > 1.5x 20-period average.
# Short: Bull Power < 0, Bear Power > 0, close < EMA13, volume > 1.5x 20-period average.
# Exit: Opposite Elder Ray condition or close crosses EMA13.
# Uses 1d timeframe for EMA13 trend alignment to avoid whipsaw.
# Volume confirmation filters false signals. Target: 12-37 trades/year (50-150 total over 4 years).
# Works in bull via trend-following long entries and in bear via short entries on weakness.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_elder_ray_regime_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # EMA13 for trend (6h)
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Volume average for confirmation (20-period)
    volume_s = pd.Series(volume)
    volume_ma = volume_s.rolling(window=20, min_periods=20).mean().values
    
    # Get 1d data for EMA13 trend alignment (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Calculate EMA13 on 1d
    close_1d_s = pd.Series(close_1d)
    ema13_1d = close_1d_s.ewm(span=13, adjust=False, min_periods=13).mean().values
    # Align 1d EMA13 to 6h
    ema13_1d_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 (using 6h EMA13)
    bull_power = high - ema13
    bear_power = low - ema13
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is NaN
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(ema13_1d_aligned[i]) or np.isnan(volume_ma[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        if position == 1:  # Long position
            # Exit: Bear Power > 0 (bulls weak) OR close < EMA13 (trend break)
            if bear_power[i] > 0 or close[i] < ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bull Power < 0 (bears weak) OR close > EMA13 (trend break)
            if bull_power[i] < 0 or close[i] > ema13[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Long entry: Bull Power > 0 AND Bear Power < 0 AND close > EMA13 AND volume confirmed
            if (bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema13[i] and
                volume_confirmed and close[i] > ema13_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: Bull Power < 0 AND Bear Power > 0 AND close < EMA13 AND volume confirmed
            elif (bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema13[i] and
                  volume_confirmed and close[i] < ema13_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals