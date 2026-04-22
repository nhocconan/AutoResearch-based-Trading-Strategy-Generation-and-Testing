#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams %R with 1-day trend filter and volume confirmation.
Long when Williams %R < -80 (oversold) and 1-day close > 20 EMA (uptrend) and 1-day volume > 20-day average volume.
Short when Williams %R > -20 (overbought) and 1-day close < 20 EMA (downtrend) and 1-day volume > 20-day average volume.
Exit when Williams %R crosses above -50 (for longs) or below -50 (for shorts).
Williams %R identifies overextended moves; trend filter ensures trading with higher timeframe momentum;
volume confirmation ensures institutional participation. Designed for low turnover to avoid fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Williams %R (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero when highest_high == lowest_low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Load 1-day data for trend and volume filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # 20-period EMA on 1-day close
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 20-day average volume on 1-day
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align 1-day indicators to 12h timeframe
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if np.isnan(williams_r[i]) or np.isnan(ema_20_1d_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Oversold + uptrend + high volume
            if williams_r[i] < -80 and close_1d[i] > ema_20_1d[i] and volume_1d[i] > avg_vol_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Overbought + downtrend + high volume
            elif williams_r[i] > -20 and close_1d[i] < ema_20_1d[i] and volume_1d[i] > avg_vol_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Williams %R crosses -50
            exit_signal = False
            
            if position == 1:
                # Exit long: Williams %R rises above -50
                if williams_r[i] > -50:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Williams %R falls below -50
                if williams_r[i] < -50:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsR_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0