#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversal with 12h EMA50 trend filter and volume confirmation
- Long when: Williams %R(14) crosses above -80 (oversold reversal) + price > 12h EMA50 + volume > 1.5x 20-period average
- Short when: Williams %R(14) crosses below -20 (overbought reversal) + price < 12h EMA50 + volume > 1.5x 20-period average
- Exit when: Williams %R crosses above -20 (for long) or below -80 (for short) OR opposing signal
- Williams %R captures short-term exhaustion in both bull and bear markets
- 12h EMA50 ensures trades align with intermediate trend to avoid chop
- Volume confirmation filters weak reversals
- Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag on 6h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate Williams %R(14): (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Williams %R signals: cross above -80 (long), cross below -20 (short)
    williams_r_long_signal = (williams_r > -80) & (np.concatenate([[True], williams_r[:-1] <= -80]))
    williams_r_short_signal = (williams_r < -20) & (np.concatenate([[True], williams_r[:-1] >= -20]))
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 1.5 * vol_ma
    
    # Load 12h EMA50 ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 14, 50)  # Need 20 for volume MA, 14 for Williams %R, 50 for EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 + price > 12h EMA50 + volume spike
            if williams_r_long_signal[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 + price < 12h EMA50 + volume spike
            elif williams_r_short_signal[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -20 OR short signal
            if williams_r[i] > -20 or williams_r_short_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -80 OR long signal
            if williams_r[i] < -80 or williams_r_long_signal[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WilliamsR_Extreme_12hEMA50_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0