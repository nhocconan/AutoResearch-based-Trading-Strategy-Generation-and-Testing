#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R with 1d trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) and price above 1d EMA50 with above-average volume.
Short when Williams %R > -20 (overbought) and price below 1d EMA50 with above-average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Williams %R identifies exhaustion points; trend filter ensures direction alignment; volume confirms institutional participation.
Works in both bull and bear markets by capturing reversals at extremes with trend and volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    
    # Load 1-day data for trend filter and volume confirmation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1-day average volume for confirmation
    volume_1d = df_1d['volume'].values
    avg_vol_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold + above 1d EMA50 + above-average volume
            if williams_r[i] < -80 and close[i] > ema_50_1d_aligned[i] and volume[i] > avg_vol_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + below 1d EMA50 + above-average volume
            elif williams_r[i] > -20 and close[i] < ema_50_1d_aligned[i] and volume[i] > avg_vol_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R crosses above -50 (recovering from oversold)
                if williams_r[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R crosses below -50 (declining from overbought)
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsR_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0