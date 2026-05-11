# 6h_LongTermTrend_WithVolumeConfirmation
# Uses 12h EMA for long-term trend and volume spike for entry confirmation
# Designed to capture major trend moves with low trade frequency to avoid fee drag
# Works in both bull and bear markets by following the trend direction
# Target: 12-30 trades per year (50-120 over 4 years)

#!/usr/bin/env python3
name = "6h_LongTermTrend_WithVolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # 12h EMA50 for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume spike detection: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after indicators are ready
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if trend data is not ready
        if np.isnan(ema50_12h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Look for volume spike in direction of trend
            if volume_spike[i]:
                if close[i] > ema50_12h_aligned[i]:
                    # Uptrend + volume spike = long
                    signals[i] = 0.25
                    position = 1
                elif close[i] < ema50_12h_aligned[i]:
                    # Downtrend + volume spike = short
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Exit long when trend turns down
            if close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when trend turns up
            if close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals